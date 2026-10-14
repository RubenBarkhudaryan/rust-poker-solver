// Solves one spot with the OpenCL solver and/or the CPU solver and compares them.
//
// Usage: cargo run --release --features opencl --example opencl -- [turn|flop|hard] [gpu|cpu|both] [max_iterations] [exact|fast]
//
// turn and flop solve a single raised pot on QsJh2h with narrow ranges; hard solves a button vs big blind single
// raised pot on Th9h4c with full ranges, 100 big blinds deep and two bet sizes per street.
//
// Both solvers check the exploitability every 10 iterations and stop at 0.5% of the pot.

use postflop_solver::*;
use std::time::Instant;

const OOP_RANGE: &str = "QQ:0.5,JJ:0.75,TT,99,88,77,66,55,44,33,22,AKo:0.25,AQs,AQo:0.75,AJs,AJo:0.75,ATs,\
ATo:0.75,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQ,KJ,KTs,KTo:0.5,K9s,K8s,K7s,K6s,K5s,K4s:0.5,K3s:0.5,K2s:0.5,QJ,\
QTs,Q9s,Q8s,Q7s,JTs,JTo:0.5,J9s,J8s,T9s,T8s,T7s,98s,97s,96s,87s,86s,76s,75s,65s,64s,54s,53s,43s";

const IP_RANGE: &str = "AA,KK,QQ,JJ,TT,99:0.75,88:0.75,77:0.5,66:0.25,55:0.25,AK,AQs,AQo:0.75,AJs,AJo:0.5,\
ATs:0.75,A6s:0.25,A5s:0.75,A4s:0.75,A3s:0.5,A2s:0.5,KQs,KQo:0.5,KJs,KTs:0.75,K5s:0.25,K4s:0.25,QJs:0.75,\
QTs:0.75,Q9s:0.5,JTs:0.75,J9s:0.75,J8s:0.75,T9s:0.75,T8s:0.75,T7s:0.75,98s:0.75,97s:0.75,96s:0.5,87s:0.75,\
86s:0.5,85s:0.5,76s:0.75,75s:0.5,65s:0.75,64s:0.5,54s:0.75,53s:0.5,43s:0.5";

// button open vs big blind call, single raised pot
const HARD_OOP_RANGE: &str =
    "22+,A2s+,K2s+,Q2s+,J4s+,T6s+,96s+,85s+,75s+,64s+,53s+,43s,A2o+,K7o+,Q8o+,J8o+,T8o+,98o";
const HARD_IP_RANGE: &str =
    "22+,A2s+,K2s+,Q3s+,J6s+,T6s+,96s+,85s+,75s+,64s+,54s,A4o+,K9o+,Q9o+,J9o+,T9o";

struct Spot {
    game: PostFlopGame,
    starting_pot: i32,
}

fn build_game(spot: &str) -> Spot {
    let (card_config, tree_config) = if spot == "hard" {
        let card_config = CardConfig {
            range: [
                HARD_OOP_RANGE.parse().unwrap(),
                HARD_IP_RANGE.parse().unwrap(),
            ],
            flop: flop_from_str("Th9h4c").unwrap(),
            turn: NOT_DEALT,
            river: NOT_DEALT,
        };
        let flop = BetSizeOptions::try_from(("33%, 75%", "3x")).unwrap();
        let turn = BetSizeOptions::try_from(("66%", "3x")).unwrap();
        let river = BetSizeOptions::try_from(("75%", "3x")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::Flop,
            starting_pot: 55,
            effective_stack: 972,
            rake_rate: 0.0,
            rake_cap: 0.0,
            flop_bet_sizes: [flop.clone(), flop],
            turn_bet_sizes: [turn.clone(), turn],
            river_bet_sizes: [river.clone(), river],
            turn_donk_sizes: None,
            river_donk_sizes: None,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.1,
        };
        (card_config, tree_config)
    } else {
        let turn = spot != "flop";
        let card_config = CardConfig {
            range: [OOP_RANGE.parse().unwrap(), IP_RANGE.parse().unwrap()],
            flop: flop_from_str("QsJh2h").unwrap(),
            turn: if turn {
                card_from_str("8d").unwrap()
            } else {
                NOT_DEALT
            },
            river: NOT_DEALT,
        };
        let sizes = BetSizeOptions::try_from(("50%, 100%", "60%")).unwrap();
        let tree_config = TreeConfig {
            initial_state: if turn {
                BoardState::Turn
            } else {
                BoardState::Flop
            },
            starting_pot: 10,
            effective_stack: 95,
            rake_rate: 0.0,
            rake_cap: 0.0,
            flop_bet_sizes: [sizes.clone(), sizes.clone()],
            turn_bet_sizes: [sizes.clone(), sizes.clone()],
            river_bet_sizes: [sizes.clone(), sizes],
            turn_donk_sizes: None,
            river_donk_sizes: None,
            add_allin_threshold: 0.0,
            force_allin_threshold: 0.67,
            merging_threshold: 0.0,
        };
        (card_config, tree_config)
    };

    let starting_pot = tree_config.starting_pot;
    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game.allocate_memory(false);
    Spot { game, starting_pot }
}

fn percent(exploitability: f32, starting_pot: i32) -> f32 {
    exploitability / starting_pot as f32 * 100.0
}

struct RunResult {
    iterations: u32,
    exploitability: f32,
    /// time spent in solver iterations
    solve: f64,
    /// time spent computing the exploitability
    check: f64,
}

/// Runs `step` until the target exploitability or `max_iterations`, checking every 10 iterations.
fn run(
    max_iterations: u32,
    starting_pot: i32,
    mut step: impl FnMut(u32),
    mut exploitability: impl FnMut() -> f32,
) -> RunResult {
    let target = starting_pot as f32 * 0.005;
    let mut result = RunResult {
        iterations: 0,
        exploitability: 0.0,
        solve: 0.0,
        check: 0.0,
    };
    let mut timed_exploitability = |result: &mut RunResult| {
        let start = Instant::now();
        result.exploitability = exploitability();
        result.check += start.elapsed().as_secs_f64();
    };
    timed_exploitability(&mut result);
    while result.iterations < max_iterations && result.exploitability > target {
        let start = Instant::now();
        step(result.iterations);
        result.solve += start.elapsed().as_secs_f64();
        result.iterations += 1;
        let t = result.iterations;
        if t % 10 == 0 || t == max_iterations {
            timed_exploitability(&mut result);
            println!(
                "  iteration {t:4}  {:8.3} s  exploitability {:.3}%",
                result.solve + result.check,
                percent(result.exploitability, starting_pot)
            );
        }
    }
    result
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let spot = args.get(1).map_or("turn", |s| s.as_str());
    let which = args.get(2).map_or("both", |s| s.as_str());
    let max_iterations = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let precision = match args.get(4).map(|s| s.as_str()) {
        Some("fast") => OpenClPrecision::Fast,
        _ => OpenClPrecision::Exact,
    };

    let mut results = Vec::new();

    if which != "cpu" {
        let start = Instant::now();
        let Spot {
            mut game,
            starting_pot,
        } = build_game(spot);
        let build = start.elapsed().as_secs_f64();

        let mut solver = OpenClSolver::new(&game, precision).unwrap();
        println!(
            "OpenCL on {} ({}): {} nodes in {} levels, {:.1} MB of device memory",
            solver.device_name(),
            if solver.is_exact() { "exact" } else { "fast" },
            solver.num_nodes(),
            solver.num_levels(),
            solver.memory_usage() as f64 / 1e6
        );

        let solver = std::cell::RefCell::new(&mut solver);
        let result = run(
            max_iterations,
            starting_pot,
            |t| {
                let mut solver = solver.borrow_mut();
                solver.solve_step(t).unwrap();
                solver.finish().unwrap();
            },
            || solver.borrow_mut().compute_exploitability().unwrap(),
        );

        let start = Instant::now();
        solver.borrow().write_back(&mut game).unwrap();
        let download = start.elapsed().as_secs_f64();
        let start = Instant::now();
        finalize(&mut game);
        let finalize_time = start.elapsed().as_secs_f64();

        let setup = solver.borrow().setup_times();
        let storage_mb = solver.borrow().storage_bytes() as f64 / 1e6;
        let mb = setup.upload_bytes as f64 / 1e6;
        let up = setup.upload.as_secs_f64();
        let iterations = result.iterations.max(1) as f64;
        println!();
        println!("time breakdown (OpenCL):");
        println!("  build the game tree (CPU)          {build:8.3} s");
        println!(
            "  lay out the tree for the GPU (CPU) {:8.3} s",
            setup.layout.as_secs_f64()
        );
        println!(
            "  compile the kernels                {:8.3} s",
            setup.compile.as_secs_f64()
        );
        println!(
            "  upload to the GPU                  {up:8.3} s  ({mb:.0} MB, {:.1} GB/s)",
            mb / 1e3 / up
        );
        println!(
            "  solve: {} iterations              {:8.3} s  ({:.1} ms per iteration)",
            result.iterations,
            result.solve,
            result.solve / iterations * 1e3
        );
        println!(
            "  exploitability checks              {:8.3} s",
            result.check
        );
        println!(
            "  download from the GPU              {download:8.3} s  ({:.0} MB, {:.1} GB/s)",
            storage_mb,
            storage_mb / 1e3 / download
        );
        println!("  finalize (CPU)                     {finalize_time:8.3} s");
        results.push(("OpenCL", result, game, starting_pot));
    }

    if which != "gpu" {
        let Spot {
            mut game,
            starting_pot,
        } = build_game(spot);
        let (mem, _) = game.memory_usage();
        println!(
            "CPU with {} threads, {:.1} MB of memory",
            rayon::current_num_threads(),
            mem as f64 / 1e6
        );
        let result = run(
            max_iterations,
            starting_pot,
            |t| solve_step(&game, t),
            || compute_exploitability(&game),
        );
        println!(
            "  solve {:.3} s ({:.1} ms per iteration), exploitability checks {:.3} s",
            result.solve,
            result.solve / result.iterations.max(1) as f64 * 1e3,
            result.check
        );
        finalize(&mut game);
        results.push(("CPU", result, game, starting_pot));
    }

    println!();
    for (name, result, _, starting_pot) in &results {
        println!(
            "{name:>6}: {} iterations, {:.3} s, exploitability {:.3}% of the pot",
            result.iterations,
            result.solve + result.check,
            percent(result.exploitability, *starting_pot)
        );
    }

    // The two solvers may not stop at the same iteration, so this only shows they agree on the root strategy.
    if let [(_, _, a, _), (_, _, b, _)] = &mut results[..] {
        let (sa, sb) = (a.strategy(), b.strategy());
        let max_diff = sa
            .iter()
            .zip(&sb)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        println!("root strategy: largest difference between the two solvers {max_diff:.4}");
    }
}
