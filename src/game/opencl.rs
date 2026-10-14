//! Discounted CFR on an OpenCL device.
//!
//! The game tree is laid out by depth: every node of `node_arena` becomes one instance in the level of its
//! depth, and each level holds one row of `H` floats per instance. An iteration walks the levels top-down to
//! push the opponent's reach probabilities, then bottom-up to compute counterfactual values and update the
//! traverser's regrets and cumulative strategy, with one kernel launch per node type and level. The strategy
//! and regrets live in device buffers that mirror `storage1` and `storage2` byte for byte, so the result can
//! be copied back and the rest of the library (`finalize`, the result interpreter, saving) works unchanged.

use super::*;
use crate::interface::*;
use crate::solver::DiscountParams;
use crate::utility::finalize;
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use opencl3::types::CL_BLOCKING;
use std::collections::HashMap;
use std::ffi::c_void;
use std::io::{self, Write};
use std::mem;
use std::ptr;
use std::time::{Duration, Instant};

const KERNEL_SOURCE: &str = include_str!("opencl_kernels.cl");

/// The floating point precision of the terminal node evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OpenClPrecision {
    /// Sums in 64-bit floats, like the CPU solver, with which the results agree bit for bit. Falls back to
    /// `Fast` on devices without 64-bit float support.
    #[default]
    Exact,
    /// Sums in 32-bit floats. Faster on GPUs with slow 64-bit floats (most consumer GPUs); the results differ
    /// from the CPU solver's by rounding, which the solving process can amplify.
    Fast,
}

/// Work-group size of the terminal kernel.
const WG: usize = 128;

/// Work-group width of the per-hand kernels.
const ROW_WG: usize = 64;

const MODE_CFR: i32 = 0;
const MODE_BEST_RESPONSE: i32 = 1;
const MODE_AVERAGE: i32 = 2;

// Mirrors of the structs in `opencl_kernels.cl`.

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ActionDesc {
    inst: i32,
    owner: i32,
    num_actions: i32,
    child_first: i32,
    offset: u32,
    pad: [i32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ChanceDesc {
    inst: i32,
    first: i32,
    count: i32,
    child_first: i32,
    num_children: i32,
    factor: f32,
    pad: [i32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Contrib {
    child: i32,
    perm: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TermDesc {
    inst: i32,
    kind: i32,
    folded: i32,
    turn: i32,
    river: i32,
    board: i32,
    win: f32,
    lose: f32,
    tie: f32,
    pad: f32,
    win_f64: f64,
    lose_f64: f64,
    tie_f64: f64,
}

/// Host side description of one depth level, before upload.
#[derive(Default)]
struct LevelPlan {
    num_inst: usize,
    actions: Vec<ActionDesc>,
    chances: Vec<ChanceDesc>,
    contribs: Vec<Contrib>,
    terminals: Vec<TermDesc>,
}

/// One depth level on the device. Descriptor buffers are absent when the level has no such nodes, since
/// OpenCL does not allow empty buffers.
struct Level {
    num_inst: usize,
    vec: Buffer<f32>,
    actions: Option<(Buffer<ActionDesc>, usize)>,
    chances: Option<(Buffer<ChanceDesc>, usize, Buffer<Contrib>)>,
    terminals: Option<(Buffer<TermDesc>, usize)>,
}

/// Walks `node_arena` and assigns every node to its depth level.
struct Planner<'a> {
    game: &'a PostFlopGame,
    levels: Vec<LevelPlan>,
    board_ids: HashMap<usize, i32>,
    boards: Vec<usize>,
    perm_ids: HashMap<usize, i32>,
    perms: Vec<&'a [Vec<(u16, u16)>; 2]>,
    max_actions: usize,
}

impl<'a> Planner<'a> {
    fn allocate(&mut self, depth: usize, count: usize) -> usize {
        if self.levels.len() <= depth {
            self.levels.resize_with(depth + 1, Default::default);
        }
        let level = &mut self.levels[depth];
        let first = level.num_inst;
        level.num_inst += count;
        first
    }

    fn board_id(&mut self, pair_index: usize) -> i32 {
        let next = self.boards.len() as i32;
        *self.board_ids.entry(pair_index).or_insert_with(|| {
            self.boards.push(pair_index);
            next
        })
    }

    fn perm_id(&mut self, swap: &'a [Vec<(u16, u16)>; 2]) -> i32 {
        let next = self.perms.len() as i32;
        *self
            .perm_ids
            .entry(swap as *const _ as usize)
            .or_insert_with(|| {
                self.perms.push(swap);
                next
            })
    }

    fn expand(&mut self, node_index: usize, depth: usize, inst: usize) {
        let game = self.game;
        let node = game.node_arena[node_index].lock();

        if node.is_terminal() {
            let config = &game.tree_config;
            let pot = (config.starting_pot + 2 * node.amount) as f64;
            let half_pot = 0.5 * pot;
            let rake = (pot * config.rake_rate).min(config.rake_cap);
            let is_fold = node.player & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG;
            let board = if is_fold {
                -1
            } else {
                self.board_id(card_pair_to_index(node.turn, node.river))
            };
            let num_combinations = game.num_combinations;
            let win = (half_pot - rake) / num_combinations;
            let lose = -half_pot / num_combinations;
            let tie = if rake == 0.0 {
                0.0
            } else {
                -0.5 * rake / num_combinations
            };
            self.levels[depth].terminals.push(TermDesc {
                inst: inst as i32,
                kind: if is_fold { 0 } else { 1 },
                folded: (node.player & PLAYER_MASK) as i32,
                turn: node.turn as i32,
                river: node.river as i32,
                board,
                win: win as f32,
                lose: lose as f32,
                tie: tie as f32,
                pad: 0.0,
                win_f64: win,
                lose_f64: lose,
                tie_f64: tie,
            });
            return;
        }

        let num_actions = node.num_actions();
        let first = self.allocate(depth + 1, num_actions);
        let child_base = node_index + node.children_offset as usize;

        if node.is_chance() {
            let factor = 1.0 / game.chance_factor(&node) as f32;
            let contrib_first = self.levels[depth].contribs.len();
            for action in 0..num_actions {
                self.levels[depth].contribs.push(Contrib {
                    child: (first + action) as i32,
                    perm: -1,
                });
            }
            for (i, &isomorphic_index) in game.isomorphic_chances(&node).iter().enumerate() {
                let perm = self.perm_id(game.isomorphic_swap(&node, i));
                self.levels[depth].contribs.push(Contrib {
                    child: (first + isomorphic_index as usize) as i32,
                    perm,
                });
            }
            let count = self.levels[depth].contribs.len() - contrib_first;
            self.levels[depth].chances.push(ChanceDesc {
                inst: inst as i32,
                first: contrib_first as i32,
                count: count as i32,
                child_first: first as i32,
                num_children: num_actions as i32,
                factor,
                pad: [0; 2],
            });
        } else {
            let offset = (node.storage1 as usize - game.storage1.as_ptr() as usize) / 4;
            self.max_actions = self.max_actions.max(num_actions);
            self.levels[depth].actions.push(ActionDesc {
                inst: inst as i32,
                owner: node.player as i32,
                num_actions: num_actions as i32,
                child_first: first as i32,
                offset: offset as u32,
                pad: [0; 3],
            });
        }

        for action in 0..num_actions {
            self.expand(child_base + action, depth + 1, first + action);
        }
    }
}

/// How long the steps of [`OpenClSolver::new`] took.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenClSetupTimes {
    /// Laying out the tree by depth and building the lookup tables, on the CPU.
    pub layout: Duration,
    /// Creating the OpenCL context and compiling the kernels.
    pub compile: Duration,
    /// Allocating the device buffers and copying the tables, strategy and regrets to the device.
    pub upload: Duration,
    /// The number of bytes copied to the device.
    pub upload_bytes: u64,
}

/// A Discounted CFR solver that runs a [`PostFlopGame`] on an OpenCL device.
///
/// The game must have its memory allocated without compression; bunching effect and node locking are not
/// supported. The solver works on its own copy of the strategy: call [`write_back`] to copy it into the
/// game, or use [`solve_opencl`], which does that and finalizes the game.
///
/// [`write_back`]: OpenClSolver::write_back
pub struct OpenClSolver {
    device_name: String,
    is_exact: bool,
    setup_times: OpenClSetupTimes,
    memory_usage: u64,
    num_hands: [usize; 2],
    h: usize,
    is_raked: bool,
    initial_weights: [Vec<f32>; 2],
    num_storage: usize,
    levels: Vec<Level>,
    strategy: Buffer<u8>,
    regrets: Buffer<u8>,
    weights: Buffer<f32>,
    cards: Buffer<i32>,
    same: Buffer<i32>,
    card_off: Buffer<i32>,
    card_list: Buffer<i32>,
    strength: Buffer<u16>,
    sorted: Buffer<u16>,
    count: Buffer<i32>,
    lt: Buffer<u16>,
    le: Buffer<u16>,
    card_sorted: Buffer<u16>,
    card_start: Buffer<i32>,
    card_pos: Buffer<u16>,
    perms: Buffer<i32>,
    k_push_action: Kernel,
    k_push_chance: Kernel,
    k_action: Kernel,
    k_chance: Kernel,
    k_terminal: Kernel,
    queue: CommandQueue,
    _program: Program,
    _context: Context,
}

fn cl_err<E: std::fmt::Display>(what: &str) -> impl FnOnce(E) -> String + '_ {
    move |e| format!("OpenCL {what} failed: {e}")
}

/// Picks the GPU with the most global memory.
fn select_device() -> Result<Device, String> {
    let platforms = get_platforms().map_err(|e| {
        if e.to_string().contains("DLOPEN") {
            "the OpenCL library (OpenCL.dll / libOpenCL.so) could not be loaded: install a GPU driver with \
             OpenCL support (on WSL, NVIDIA's driver provides CUDA but not OpenCL, so run the Windows build)"
                .to_string()
        } else {
            format!("OpenCL platform query failed: {e}")
        }
    })?;
    let mut best: Option<(u64, Device)> = None;
    for platform in platforms {
        let Ok(ids) = platform.get_devices(CL_DEVICE_TYPE_GPU) else {
            continue;
        };
        for id in ids {
            let device = Device::new(id);
            let memory = device.global_mem_size().unwrap_or(0);
            if best.as_ref().is_none_or(|(m, _)| memory > *m) {
                best = Some((memory, device));
            }
        }
    }
    best.map(|(_, d)| d)
        .ok_or_else(|| "no OpenCL GPU device found".to_string())
}

impl OpenClSolver {
    /// Lays out the tree of `game` on the best available GPU and uploads its current strategy and regrets.
    pub fn new(game: &PostFlopGame, precision: OpenClPrecision) -> Result<Self, String> {
        if !game.is_ready() {
            return Err("game is not ready: allocate its memory first".to_string());
        }
        if game.is_compression_enabled {
            return Err("the OpenCL solver does not support compression".to_string());
        }
        if game.bunching_num_dead_cards != 0 {
            return Err("the OpenCL solver does not support the bunching effect".to_string());
        }
        if !game.locking_strategy.is_empty() {
            return Err("the OpenCL solver does not support node locking".to_string());
        }

        let start = Instant::now();
        let num_hands = [game.private_cards[0].len(), game.private_cards[1].len()];
        let h = num_hands[0].max(num_hands[1]);
        let num_storage = game.storage1.len() / 4;
        if num_storage > u32::MAX as usize {
            return Err("the game tree is too large for 32-bit storage offsets".to_string());
        }

        // lay out the tree by depth
        let mut planner = Planner {
            game,
            levels: vec![LevelPlan::default()],
            board_ids: HashMap::new(),
            boards: Vec::new(),
            perm_ids: HashMap::new(),
            perms: Vec::new(),
            max_actions: 1,
        };
        planner.allocate(0, 1);
        planner.expand(0, 0, 0);

        // hand tables
        let mut cards = vec![0i32; 2 * 2 * h];
        let mut same = vec![-1i32; 2 * h];
        let mut card_off = vec![0i32; 2 * 53];
        let mut card_list = Vec::new();
        for player in 0..2 {
            for (i, &(c1, c2)) in game.private_cards[player].iter().enumerate() {
                cards[2 * (player * h + i)] = c1 as i32;
                cards[2 * (player * h + i) + 1] = c2 as i32;
                let s = game.same_hand_index[player][i];
                same[player * h + i] = if s == u16::MAX { -1 } else { s as i32 };
            }
            for card in 0..52u8 {
                card_off[player * 53 + card as usize] = card_list.len() as i32;
                for (i, &(c1, c2)) in game.private_cards[player].iter().enumerate() {
                    if c1 == card || c2 == card {
                        card_list.push(i as i32);
                    }
                }
            }
            card_off[player * 53 + 52] = card_list.len() as i32;
        }

        // river board tables
        let num_boards = planner.boards.len().max(1);
        let mut strength = vec![0u16; num_boards * 2 * h];
        let mut sorted = vec![0u16; num_boards * 2 * h];
        let mut count = vec![0i32; num_boards * 2];
        let mut lt = vec![0u16; num_boards * 2 * h];
        let mut le = vec![0u16; num_boards * 2 * h];
        for (board, &pair_index) in planner.boards.iter().enumerate() {
            let items = &game.hand_strength[pair_index];
            let valid = |player: usize| &items[player][1..items[player].len() - 1];
            for player in 0..2 {
                let base = (board * 2 + player) * h;
                for (k, item) in valid(player).iter().enumerate() {
                    strength[base + item.index as usize] = item.strength;
                    sorted[base + k] = item.index;
                }
                count[board * 2 + player] = valid(player).len() as i32;
            }
            for player in 0..2 {
                let base = (board * 2 + player) * h;
                let opponent = valid(player ^ 1);
                let (mut below, mut at_most) = (0, 0);
                for item in valid(player) {
                    while below < opponent.len() && opponent[below].strength < item.strength {
                        below += 1;
                    }
                    at_most = at_most.max(below);
                    while at_most < opponent.len() && opponent[at_most].strength <= item.strength {
                        at_most += 1;
                    }
                    lt[base + item.index as usize] = below as u16;
                    le[base + item.index as usize] = at_most as u16;
                }
            }
        }

        // the same hands grouped by card, for removing the hands that share a card with ours
        let cs = 2 * h + 52;
        let mut card_sorted = vec![0u16; num_boards * 2 * cs];
        let mut card_start = vec![0i32; num_boards * 2 * 53];
        let mut card_pos = vec![0u16; num_boards * 2 * h * 4];
        for (board, &pair_index) in planner.boards.iter().enumerate() {
            let items = &game.hand_strength[pair_index];
            let valid = |player: usize| &items[player][1..items[player].len() - 1];
            for player in 0..2 {
                // groups of the list owner's hands, in ascending strength order
                let mut groups: [Vec<&StrengthItem>; 52] = std::array::from_fn(|_| Vec::new());
                for item in valid(player) {
                    let (c1, c2) = game.private_cards[player][item.index as usize];
                    groups[c1 as usize].push(item);
                    groups[c2 as usize].push(item);
                }
                let list = board * 2 + player;
                let mut offset = 0;
                for (card, group) in groups.iter().enumerate() {
                    card_start[list * 53 + card] = offset as i32;
                    for (k, item) in group.iter().enumerate() {
                        card_sorted[list * cs + offset + k] = item.index;
                    }
                    offset += group.len() + 1;
                }
                card_start[list * 53 + 52] = offset as i32;

                // positions of the other player's hands in those groups
                let other = player ^ 1;
                let base = (board * 2 + other) * h;
                for item in valid(other) {
                    let (c1, c2) = game.private_cards[other][item.index as usize];
                    let pos = &mut card_pos[(base + item.index as usize) * 4..][..4];
                    for (k, card) in [c1, c2].into_iter().enumerate() {
                        let group = &groups[card as usize];
                        pos[2 * k] = group.partition_point(|g| g.strength < item.strength) as u16;
                        pos[2 * k + 1] =
                            group.partition_point(|g| g.strength <= item.strength) as u16;
                    }
                }
            }
        }

        // hand permutations of isomorphic chances
        let mut perms = vec![0i32; planner.perms.len().max(1) * 2 * h];
        for (id, swap) in planner.perms.iter().enumerate() {
            for player in 0..2 {
                let row = &mut perms[(id * 2 + player) * h..(id * 2 + player + 1) * h];
                row.iter_mut().enumerate().for_each(|(i, p)| *p = i as i32);
                for &(i, j) in &swap[player] {
                    row[i as usize] = j as i32;
                    row[j as usize] = i as i32;
                }
            }
        }

        let mut weights = vec![0f32; 2 * h];
        for player in 0..2 {
            weights[player * h..player * h + num_hands[player]]
                .copy_from_slice(&game.initial_weights[player]);
        }

        let layout = start.elapsed();

        // device and memory check
        let device = select_device()?;
        let device_name = device.name().map_err(cl_err("device query"))?;
        let global_memory = device.global_mem_size().map_err(cl_err("device query"))?;
        let max_alloc = device
            .max_mem_alloc_size()
            .map_err(cl_err("device query"))?;

        let vec_bytes = |num_inst: usize| (num_inst * h * 4) as u64;
        let largest_level = planner.levels.iter().map(|l| vec_bytes(l.num_inst)).max();
        let storage_bytes = (num_storage * 4) as u64;
        let largest = largest_level.unwrap_or(0).max(storage_bytes);
        let descriptor_bytes: u64 = planner
            .levels
            .iter()
            .map(|l| {
                (l.actions.len() * mem::size_of::<ActionDesc>()
                    + l.chances.len() * mem::size_of::<ChanceDesc>()
                    + l.contribs.len() * mem::size_of::<Contrib>()
                    + l.terminals.len() * mem::size_of::<TermDesc>()) as u64
            })
            .sum();
        let table_bytes = ((cards.len() + same.len() + card_off.len() + card_list.len()) * 4
            + (strength.len()
                + sorted.len()
                + lt.len()
                + le.len()
                + card_sorted.len()
                + card_pos.len())
                * 2
            + card_start.len() * 4
            + (count.len() + perms.len() + weights.len()) * 4) as u64;
        let memory_usage = planner
            .levels
            .iter()
            .map(|l| vec_bytes(l.num_inst))
            .sum::<u64>()
            + 2 * storage_bytes
            + descriptor_bytes
            + table_bytes;

        if largest > max_alloc {
            return Err(format!(
                "a buffer of {} MB exceeds the device's maximum allocation of {} MB",
                largest >> 20,
                max_alloc >> 20
            ));
        }
        if memory_usage > global_memory / 20 * 19 {
            return Err(format!(
                "the solver needs {} MB but {} has {} MB",
                memory_usage >> 20,
                device_name,
                global_memory >> 20
            ));
        }

        let start = Instant::now();
        let context = Context::from_device(&device).map_err(cl_err("context creation"))?;
        #[allow(deprecated)]
        let queue = CommandQueue::create_default(&context, 0).map_err(cl_err("queue creation"))?;
        // correctly rounded division keeps regret matching identical to the CPU
        let mut options = format!("-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt -D H={h} -D WG={WG} -D CS={cs} -D MAXA={}", planner.max_actions);
        let extensions = device.extensions().map_err(cl_err("device query"))?;
        let is_exact = precision == OpenClPrecision::Exact && extensions.contains("cl_khr_fp64");
        if !is_exact {
            options.push_str(" -D NO_FP64");
        }
        let program = Program::create_and_build_from_source(&context, KERNEL_SOURCE, &options)
            .map_err(|log| format!("OpenCL kernel build failed: {log}"))?;
        let kernel = |name: &str| Kernel::create(&program, name).map_err(cl_err("kernel creation"));

        // read-only uploads; the host data is copied at creation, so the vectors can be dropped afterwards
        fn upload<T>(context: &Context, data: &[T], flags: u64) -> Result<Buffer<T>, String> {
            unsafe {
                Buffer::<T>::create(
                    context,
                    flags | CL_MEM_COPY_HOST_PTR,
                    data.len(),
                    data.as_ptr() as *mut c_void,
                )
            }
            .map_err(cl_err("buffer creation"))
        }
        fn upload_opt<T>(context: &Context, data: &[T]) -> Result<Option<Buffer<T>>, String> {
            if data.is_empty() {
                Ok(None)
            } else {
                upload(context, data, CL_MEM_READ_ONLY).map(Some)
            }
        }

        let compile = start.elapsed();

        let start = Instant::now();
        let mut levels = Vec::with_capacity(planner.levels.len());
        for plan in &planner.levels {
            let vec = unsafe {
                Buffer::<f32>::create(
                    &context,
                    CL_MEM_READ_WRITE,
                    plan.num_inst * h,
                    ptr::null_mut(),
                )
            }
            .map_err(cl_err("buffer creation"))?;
            let chances = match upload_opt(&context, &plan.chances)? {
                Some(chances) => Some((
                    chances,
                    plan.chances.len(),
                    upload(&context, &plan.contribs, CL_MEM_READ_ONLY)?,
                )),
                None => None,
            };
            levels.push(Level {
                num_inst: plan.num_inst,
                vec,
                actions: upload_opt(&context, &plan.actions)?.map(|b| (b, plan.actions.len())),
                chances,
                terminals: upload_opt(&context, &plan.terminals)?
                    .map(|b| (b, plan.terminals.len())),
            });
        }

        // the strategy and regrets start from the game's current state
        let storage = |bytes: &Vec<u8>| -> Result<Buffer<u8>, String> {
            let mut buffer = unsafe {
                Buffer::<u8>::create(
                    &context,
                    CL_MEM_READ_WRITE,
                    bytes.len().max(4),
                    ptr::null_mut(),
                )
            }
            .map_err(cl_err("buffer creation"))?;
            if !bytes.is_empty() {
                unsafe { queue.enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, bytes, &[]) }
                    .map_err(cl_err("write"))?;
            }
            Ok(buffer)
        };
        let strategy = storage(&game.storage1)?;
        let regrets = storage(&game.storage2)?;

        let weights = upload(&context, &weights, CL_MEM_READ_ONLY)?;
        let tables = (
            upload(&context, &cards, CL_MEM_READ_ONLY)?,
            upload(&context, &same, CL_MEM_READ_ONLY)?,
            upload(&context, &card_off, CL_MEM_READ_ONLY)?,
            upload(&context, &card_list, CL_MEM_READ_ONLY)?,
            upload(&context, &strength, CL_MEM_READ_ONLY)?,
            upload(&context, &sorted, CL_MEM_READ_ONLY)?,
            upload(&context, &count, CL_MEM_READ_ONLY)?,
            upload(&context, &lt, CL_MEM_READ_ONLY)?,
            upload(&context, &le, CL_MEM_READ_ONLY)?,
            upload(&context, &card_sorted, CL_MEM_READ_ONLY)?,
            upload(&context, &card_start, CL_MEM_READ_ONLY)?,
            upload(&context, &card_pos, CL_MEM_READ_ONLY)?,
            upload(&context, &perms, CL_MEM_READ_ONLY)?,
        );
        queue.finish().map_err(cl_err("finish"))?;
        let setup_times = OpenClSetupTimes {
            layout,
            compile,
            upload: start.elapsed(),
            upload_bytes: 2 * storage_bytes + descriptor_bytes + table_bytes,
        };

        Ok(Self {
            device_name,
            is_exact,
            setup_times,
            memory_usage,
            num_hands,
            h,
            is_raked: game.is_raked(),
            initial_weights: game.initial_weights.clone(),
            num_storage,
            levels,
            strategy,
            regrets,
            weights,
            cards: tables.0,
            same: tables.1,
            card_off: tables.2,
            card_list: tables.3,
            strength: tables.4,
            sorted: tables.5,
            count: tables.6,
            lt: tables.7,
            le: tables.8,
            card_sorted: tables.9,
            card_start: tables.10,
            card_pos: tables.11,
            perms: tables.12,
            k_push_action: kernel("push_action")?,
            k_push_chance: kernel("push_chance")?,
            k_action: kernel("action_up")?,
            k_chance: kernel("chance_up")?,
            k_terminal: kernel("terminal")?,
            queue,
            _program: program,
            _context: context,
        })
    }

    /// The name of the OpenCL device.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Whether the results agree bit for bit with the CPU solver (see [`OpenClPrecision`]).
    pub fn is_exact(&self) -> bool {
        self.is_exact
    }

    /// How long creating the solver took, step by step.
    pub fn setup_times(&self) -> OpenClSetupTimes {
        self.setup_times
    }

    /// The size of the strategy and regrets, which [`write_back`](Self::write_back) copies from the device.
    pub fn storage_bytes(&self) -> u64 {
        2 * 4 * self.num_storage as u64
    }

    /// The device memory used by the solver, in bytes.
    pub fn memory_usage(&self) -> u64 {
        self.memory_usage
    }

    /// The number of depth levels of the tree.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// The total number of tree nodes.
    pub fn num_nodes(&self) -> usize {
        self.levels.iter().map(|l| l.num_inst).sum()
    }

    /// Proceeds Discounted CFR for one iteration, like [`solve_step`](crate::solve_step).
    pub fn solve_step(&mut self, current_iteration: u32) -> Result<(), String> {
        let params = DiscountParams::new(current_iteration);
        for player in 0..2 {
            self.traverse(player, MODE_CFR, &params)?;
        }
        Ok(())
    }

    /// Computes the exploitability of the average strategy, like
    /// [`compute_exploitability`](crate::compute_exploitability).
    pub fn compute_exploitability(&mut self) -> Result<f32, String> {
        let params = DiscountParams::new(0);
        let mut mes_ev = [0.0; 2];
        for (player, ev) in mes_ev.iter_mut().enumerate() {
            self.traverse(player, MODE_BEST_RESPONSE, &params)?;
            *ev = self.root_value(player)?;
        }
        if !self.is_raked {
            return Ok((mes_ev[0] + mes_ev[1]) * 0.5);
        }
        let mut current_ev = [0.0; 2];
        for (player, ev) in current_ev.iter_mut().enumerate() {
            self.traverse(player, MODE_AVERAGE, &params)?;
            *ev = self.root_value(player)?;
        }
        Ok(((mes_ev[0] - current_ev[0]) + (mes_ev[1] - current_ev[1])) * 0.5)
    }

    /// Copies the cumulative strategy and regrets from the device into `game`, which must be the game the
    /// solver was created from.
    pub fn write_back(&self, game: &mut PostFlopGame) -> Result<(), String> {
        if game.storage1.len() / 4 != self.num_storage
            || game.storage2.len() / 4 != self.num_storage
        {
            return Err("the game does not match the solver".to_string());
        }
        if self.num_storage == 0 {
            return Ok(());
        }
        for (source, target) in [
            (&self.strategy, &mut game.storage1),
            (&self.regrets, &mut game.storage2),
        ] {
            unsafe {
                self.queue
                    .enqueue_read_buffer(source, CL_BLOCKING, 0, target, &[])
            }
            .map_err(cl_err("read"))?;
        }
        Ok(())
    }

    /// One traversal for `player`: reach probabilities down, counterfactual values up.
    fn traverse(
        &mut self,
        player: usize,
        mode: i32,
        params: &DiscountParams,
    ) -> Result<(), String> {
        let opponent = player ^ 1;
        let h = self.h;
        let num_player = self.num_hands[player] as i32;
        let num_opponent = self.num_hands[opponent] as i32;
        let rows = |n: i32| (n as usize).div_ceil(ROW_WG) * ROW_WG;
        let player_arg = player as i32;

        // the root starts from the opponent's initial weights
        unsafe {
            self.queue
                .enqueue_copy_buffer(
                    &self.weights,
                    &mut self.levels[0].vec,
                    opponent * h * 4,
                    0,
                    self.num_hands[opponent] * 4,
                    &[],
                )
                .map_err(cl_err("copy"))?;
        }

        unsafe {
            for depth in 0..self.levels.len() - 1 {
                let level = &self.levels[depth];
                let child_vec = &self.levels[depth + 1].vec;
                if let Some((actions, n)) = &level.actions {
                    ExecuteKernel::new(&self.k_push_action)
                        .set_arg(&level.vec)
                        .set_arg(child_vec)
                        .set_arg(actions)
                        .set_arg(&self.regrets)
                        .set_arg(&self.strategy)
                        .set_arg(&player_arg)
                        .set_arg(&num_opponent)
                        .set_arg(&mode)
                        .set_global_work_sizes(&[rows(num_opponent), *n])
                        .set_local_work_sizes(&[ROW_WG, 1])
                        .enqueue_nd_range(&self.queue)
                        .map_err(cl_err("push kernel"))?;
                }
                if let Some((chances, n, _)) = &level.chances {
                    ExecuteKernel::new(&self.k_push_chance)
                        .set_arg(&level.vec)
                        .set_arg(child_vec)
                        .set_arg(chances)
                        .set_arg(&num_opponent)
                        .set_global_work_sizes(&[rows(num_opponent), *n])
                        .set_local_work_sizes(&[ROW_WG, 1])
                        .enqueue_nd_range(&self.queue)
                        .map_err(cl_err("push kernel"))?;
                }
            }

            for depth in (0..self.levels.len()).rev() {
                let level = &self.levels[depth];
                if let Some((terminals, n)) = &level.terminals {
                    ExecuteKernel::new(&self.k_terminal)
                        .set_arg(&level.vec)
                        .set_arg(terminals)
                        .set_arg(&player_arg)
                        .set_arg(&num_player)
                        .set_arg(&num_opponent)
                        .set_arg(&self.cards)
                        .set_arg(&self.same)
                        .set_arg(&self.card_off)
                        .set_arg(&self.card_list)
                        .set_arg(&self.strength)
                        .set_arg(&self.sorted)
                        .set_arg(&self.count)
                        .set_arg(&self.lt)
                        .set_arg(&self.le)
                        .set_arg(&self.card_sorted)
                        .set_arg(&self.card_start)
                        .set_arg(&self.card_pos)
                        .set_global_work_sizes(&[n * WG])
                        .set_local_work_sizes(&[WG])
                        .enqueue_nd_range(&self.queue)
                        .map_err(cl_err("terminal kernel"))?;
                }
                if let Some((chances, n, contribs)) = &level.chances {
                    ExecuteKernel::new(&self.k_chance)
                        .set_arg(&self.levels[depth + 1].vec)
                        .set_arg(&level.vec)
                        .set_arg(chances)
                        .set_arg(contribs)
                        .set_arg(&self.perms)
                        .set_arg(&player_arg)
                        .set_arg(&num_player)
                        .set_global_work_sizes(&[rows(num_player), *n])
                        .set_local_work_sizes(&[ROW_WG, 1])
                        .enqueue_nd_range(&self.queue)
                        .map_err(cl_err("chance kernel"))?;
                }
                if let Some((actions, n)) = &level.actions {
                    ExecuteKernel::new(&self.k_action)
                        .set_arg(&self.levels[depth + 1].vec)
                        .set_arg(&level.vec)
                        .set_arg(actions)
                        .set_arg(&self.regrets)
                        .set_arg(&self.strategy)
                        .set_arg(&player_arg)
                        .set_arg(&num_player)
                        .set_arg(&mode)
                        .set_arg(&params.alpha_t)
                        .set_arg(&params.beta_t)
                        .set_arg(&params.gamma_t)
                        .set_global_work_sizes(&[rows(num_player), *n])
                        .set_local_work_sizes(&[ROW_WG, 1])
                        .enqueue_nd_range(&self.queue)
                        .map_err(cl_err("action kernel"))?;
                }
            }
        }

        Ok(())
    }

    /// The weighted sum of the root's counterfactual values after a traversal for `player`.
    fn root_value(&self, player: usize) -> Result<f32, String> {
        let mut values = vec![0f32; self.num_hands[player]];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.levels[0].vec, CL_BLOCKING, 0, &mut values, &[])
                .map_err(cl_err("read"))?;
        }
        let sum = values
            .iter()
            .zip(&self.initial_weights[player])
            .fold(0.0, |sum, (&v, &w)| sum + v as f64 * w as f64);
        Ok(sum as f32)
    }

    /// Waits until all queued work is done.
    pub fn finish(&self) -> Result<(), String> {
        self.queue.finish().map_err(cl_err("finish"))
    }
}

/// Performs Discounted CFR on an OpenCL device until the given number of iterations or exploitability is
/// satisfied, like [`solve`](crate::solve), then copies the result into `game` and finalizes it.
///
/// Returns the exploitability of the obtained strategy, or an error when the game cannot run on the device
/// (see [`OpenClSolver`]).
pub fn solve_opencl(
    game: &mut PostFlopGame,
    max_num_iterations: u32,
    target_exploitability: f32,
    print_progress: bool,
    precision: OpenClPrecision,
) -> Result<f32, String> {
    if game.is_solved() {
        panic!("Game is already solved");
    }

    let mut solver = OpenClSolver::new(game, precision)?;
    let mut exploitability = solver.compute_exploitability()?;

    if print_progress {
        print!("iteration: 0 / {max_num_iterations} ");
        print!("(exploitability = {exploitability:.4e})");
        io::stdout().flush().unwrap();
    }

    for t in 0..max_num_iterations {
        if exploitability <= target_exploitability {
            break;
        }

        solver.solve_step(t)?;

        if (t + 1) % 10 == 0 || t + 1 == max_num_iterations {
            exploitability = solver.compute_exploitability()?;
        }

        if print_progress {
            print!("\riteration: {} / {} ", t + 1, max_num_iterations);
            print!("(exploitability = {exploitability:.4e})");
            io::stdout().flush().unwrap();
        }
    }

    if print_progress {
        println!();
        io::stdout().flush().unwrap();
    }

    solver.write_back(game)?;
    finalize(game);

    Ok(exploitability)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bet_size::*;
    use crate::range::*;
    use crate::solver::solve_step;

    fn build_game(turn: Card, river: Card) -> PostFlopGame {
        let card_config = CardConfig {
            range: [
                "QQ:0.5,JJ:0.75,TT,99,88,66,55,22,AQs,AJo:0.75,ATs,A5s,KQ,KTs,K9s,QTs,JTs,J9s,T9s,98s,87s,76s,65s"
                    .parse()
                    .unwrap(),
                "AA,KK,QQ,JJ,TT,99:0.75,77:0.5,AK,AQs,AJs,A5s:0.75,KQs,KJs,QJs:0.75,JTs:0.75,T9s,98s,87s,76s,54s"
                    .parse()
                    .unwrap(),
            ],
            flop: flop_from_str("QsJh2h").unwrap(),
            turn,
            river,
        };
        let sizes = BetSizeOptions::try_from(("50%, 100%", "60%")).unwrap();
        let initial_state = if river != NOT_DEALT {
            BoardState::River
        } else if turn != NOT_DEALT {
            BoardState::Turn
        } else {
            BoardState::Flop
        };
        let tree_config = TreeConfig {
            initial_state,
            starting_pot: 10,
            effective_stack: 95,
            flop_bet_sizes: [sizes.clone(), sizes.clone()],
            turn_bet_sizes: [sizes.clone(), sizes.clone()],
            river_bet_sizes: [sizes.clone(), sizes],
            force_allin_threshold: 0.67,
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
        game.allocate_memory(false);
        game
    }

    fn floats(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect()
    }

    /// Largest absolute difference, relative to the largest absolute value.
    fn relative_diff(a: &[u8], b: &[u8]) -> f32 {
        let (a, b) = (floats(a), floats(b));
        let scale = a.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30);
        let diff = a
            .iter()
            .zip(&b)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        diff / scale
    }

    fn compare(turn: Card, river: Card, iterations: u32) {
        let cpu = build_game(turn, river);
        let mut gpu = build_game(turn, river);
        let mut solver = match OpenClSolver::new(&gpu, OpenClPrecision::Exact) {
            Ok(solver) => solver,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };
        for t in 0..iterations {
            solve_step(&cpu, t);
            solver.solve_step(t).unwrap();
        }
        solver.write_back(&mut gpu).unwrap();

        let strategy = relative_diff(&cpu.storage1, &gpu.storage1);
        let regrets = relative_diff(&cpu.storage2, &gpu.storage2);
        let e_cpu = crate::utility::compute_exploitability(&cpu);
        let e_gpu = solver.compute_exploitability().unwrap();
        println!(
            "turn {turn} river {river}, {iterations} iterations: strategy {strategy:.2e}, regrets {regrets:.2e}, \
             exploitability cpu {e_cpu:.5} gpu {e_gpu:.5}"
        );
        assert!(strategy < 1e-4, "strategy differs by {strategy}");
        assert!(regrets < 1e-4, "regrets differ by {regrets}");
        assert!((e_cpu - e_gpu).abs() <= 1e-3 * e_cpu.abs().max(1e-3));
    }

    #[test]
    fn matches_cpu_river() {
        compare(
            card_from_str("8d").unwrap(),
            card_from_str("3c").unwrap(),
            30,
        );
    }

    #[test]
    fn matches_cpu_turn() {
        compare(card_from_str("8d").unwrap(), NOT_DEALT, 20);
    }

    #[test]
    fn matches_cpu_flop() {
        compare(NOT_DEALT, NOT_DEALT, 5);
    }
}
