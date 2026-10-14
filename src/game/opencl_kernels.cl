// Discounted CFR kernels for `PostFlopGame`. See `opencl.rs` for how the tree is laid out.
//
// Every node of the game tree is an "instance" with one row of `H` floats in the buffer of its depth level.
// On the way down the row holds the opponent's reach probabilities, on the way up the traverser's
// counterfactual values. The arithmetic mirrors `solve_recursive` in `solver.rs` and the evaluation in
// `evaluation.rs`, so the results match the CPU solver up to floating point rounding.
//
// Defined by the host: H (row stride, the larger of the two players' hand counts), WG (work-group size of the
// terminal kernel), CS (stride of the per-card tables), MAXA (largest number of actions of a node), and NO_FP64
// for the fast precision.

#pragma OPENCL FP_CONTRACT OFF

#if defined(cl_khr_fp64) && !defined(NO_FP64)
#define USE_FP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double acc_t;
#else
typedef float acc_t;
#endif

#define MODE_CFR 0
#define MODE_BEST_RESPONSE 1
#define MODE_AVERAGE 2

typedef struct {
    int inst;
    int owner;
    int num_actions;
    int child_first; // instance index of the first child in the next level; children are contiguous
    uint offset;     // storage offset of the strategy and regrets
    int pad0, pad1, pad2;
} ActionDesc;

typedef struct {
    int inst;
    int first;        // first entry in the contribution list
    int count;        // number of contributions, isomorphic ones included
    int child_first;  // instance index of the first child in the next level; children are contiguous
    int num_children; // number of dealt cards that have a child
    float factor;     // 1 / chance factor
    int pad0, pad1;
} ChanceDesc;

typedef struct {
    int child; // instance index in the next level
    int perm;  // -1, or the index of the hand permutation of an isomorphic chance
} Contrib;

// The payoffs per combination are given in both precisions; the double ones are used when the device supports
// them, which keeps the arithmetic identical to the CPU's.
typedef struct {
    int inst;
    int kind;   // 0: fold, 1: showdown
    int folded; // the player who folded
    int turn;
    int river;
    int board;  // index into the river board tables, for showdowns
    float win;
    float lose;
    float tie;
    float pad;
#ifdef USE_FP64
    double win_d;
    double lose_d;
    double tie_d;
#else
    ulong win_d;
    ulong lose_d;
    ulong tie_d;
#endif
} TermDesc;

#ifdef USE_FP64
#define WIN(d) ((d).win_d)
#define LOSE(d) ((d).lose_d)
#define TIE(d) ((d).tie_d)
#else
#define WIN(d) ((d).win)
#define LOSE(d) ((d).lose)
#define TIE(d) ((d).tie)
#endif

inline float positive(float x) {
    return x > 0.0f ? x : 0.0f;
}

// The strategy of the hand whose first entry is at `base`: by regret matching in MODE_CFR, from the normalized
// cumulative strategy in the other modes. Returns false when it is uniform, in which case `x` is left as is.
inline bool load_strategy(__global const float* regrets, __global const float* cum, uint base, int n, int num_hands,
                          int mode, float* x, float* denom) {
    float sum = 0.0f;
    for (int a = 0; a < n; a++) {
        uint index = base + (uint)(a * num_hands);
        x[a] = mode == MODE_CFR ? positive(regrets[index]) : cum[index];
        sum += x[a];
    }
    *denom = sum;
    return as_uint(sum) != 0;
}

// Top-down: action nodes give each child the reach probabilities of the opponent, weighted by the opponent's
// strategy at the opponent's nodes.
__kernel void push_action(__global const float* vec, __global float* child_vec, __global const ActionDesc* descs,
                          __global const float* regrets, __global const float* cum, int player, int num_hands,
                          int mode) {
    int h = get_global_id(0);
    int i = get_global_id(1);
    if (h >= num_hands)
        return;

    ActionDesc d = descs[i];
    int n = d.num_actions;
    float reach = vec[(size_t)d.inst * H + h];
    __global float* child = child_vec + (size_t)d.child_first * H + h;

    if (d.owner == player || n == 1) {
        for (int a = 0; a < n; a++)
            child[(size_t)a * H] = reach;
        return;
    }

    float x[MAXA];
    float denom;
    bool weighted = load_strategy(regrets, cum, d.offset + (uint)h, n, num_hands, mode, x, &denom);
    float uniform_value = 1.0f / (float)n;
    for (int a = 0; a < n; a++) {
        float s = weighted ? x[a] / denom : uniform_value;
        child[(size_t)a * H] = s * reach;
    }
}

// Top-down: chance nodes give each child the reach probabilities divided by the chance factor.
__kernel void push_chance(__global const float* vec, __global float* child_vec, __global const ChanceDesc* descs,
                          int num_hands) {
    int h = get_global_id(0);
    int i = get_global_id(1);
    if (h >= num_hands)
        return;

    ChanceDesc d = descs[i];
    float reach = vec[(size_t)d.inst * H + h] * d.factor;
    __global float* child = child_vec + (size_t)d.child_first * H + h;
    for (int a = 0; a < d.num_children; a++)
        child[(size_t)a * H] = reach;
}

// Bottom-up: action nodes. In MODE_CFR the traverser's nodes also update their cumulative strategy and regrets.
__kernel void action_up(__global const float* child_vec, __global float* vec, __global const ActionDesc* descs,
                        __global float* regrets, __global float* cum, int player, int num_hands, int mode,
                        float alpha, float beta, float gamma) {
    int h = get_global_id(0);
    int i = get_global_id(1);
    if (h >= num_hands)
        return;

    ActionDesc d = descs[i];
    __global const float* child = child_vec + (size_t)d.child_first * H + h;
    int n = d.num_actions;
    float result = child[0];

    if (n == 1) {
        // a single action is passed through, like the CPU solver does
    } else if (d.owner != player) {
        for (int a = 1; a < n; a++)
            result += child[(size_t)a * H];
    } else if (mode == MODE_BEST_RESPONSE) {
        for (int a = 1; a < n; a++) {
            float v = child[(size_t)a * H];
            result = result > v ? result : v;
        }
    } else {
        uint base = d.offset + (uint)h;
        float x[MAXA];
        float cfv[MAXA];
        float denom;
        bool weighted = load_strategy(regrets, cum, base, n, num_hands, mode, x, &denom);
        float uniform_value = 1.0f / (float)n;

        for (int a = 0; a < n; a++) {
            cfv[a] = child[(size_t)a * H];
            float s = weighted ? x[a] / denom : uniform_value;
            float v = s * cfv[a];
            result = a == 0 ? v : result + v;
        }

        if (mode == MODE_CFR) {
            for (int a = 0; a < n; a++) {
                uint index = base + (uint)(a * num_hands);
                float r = regrets[index];
                float s = weighted ? x[a] / denom : uniform_value;
                cum[index] = cum[index] * gamma + s;
                float coef = signbit(r) ? beta : alpha;
                float updated = r * coef + cfv[a];
                regrets[index] = updated - result;
            }
        }
    }

    vec[(size_t)d.inst * H + h] = result;
}

// Bottom-up: chance nodes sum their children, with isomorphic deals reusing a child with swapped hands.
__kernel void chance_up(__global const float* child_vec, __global float* vec, __global const ChanceDesc* descs,
                        __global const Contrib* contribs, __global const int* perms, int player, int num_hands) {
    int h = get_global_id(0);
    int i = get_global_id(1);
    if (h >= num_hands)
        return;

    ChanceDesc d = descs[i];
    acc_t sum = 0;
    for (int k = 0; k < d.count; k++) {
        Contrib c = contribs[d.first + k];
        int src = c.perm < 0 ? h : perms[((size_t)c.perm * 2 + player) * H + h];
        sum += (acc_t)child_vec[(size_t)c.child * H + src];
    }
    vec[(size_t)d.inst * H + h] = (float)sum;
}

// Bottom-up: terminal nodes, one work-group each. The row holds the opponent's reach on entry and the
// traverser's counterfactual values on exit. The formulas and the precision of the sums follow
// `evaluate_internal` in `evaluation.rs`.
//
// cards / same: per player with a stride of H. card_off / card_list: per player, the hands holding each card.
// The river board tables are per board and player:
// - strength (stride H): 0 for hands blocked by the board;
// - sorted (stride H): the other hands by ascending strength, with `count` entries;
// - lt / le (stride H): how many of the opponent's hands have a strength below / at most the hand's;
// - card_sorted (stride CS) and card_start (stride 53): the hands of `sorted` grouped by card, each group
//   followed by one spare slot for its total;
// - card_pos (stride H): for both cards of the hand, how many hands of the opponent's group for that card
//   have a strength below / at most the hand's.
// With prefix sums over `sorted` and over each card group, removing the opponent's hands that share a card
// with ours takes a few lookups per hand, and the per-card sums are accumulated in the CPU's order.
__kernel void terminal(__global float* vec, __global const TermDesc* descs, int player, int num_hands, int num_opp,
                       __global const int2* cards, __global const int* same, __global const int* card_off,
                       __global const int* card_list, __global const ushort* strength,
                       __global const ushort* sorted, __global const int* count, __global const ushort* lt,
                       __global const ushort* le, __global const ushort* card_sorted,
                       __global const int* card_start, __global const ushort4* card_pos) {
    __local float lreach[H];
    __local acc_t prefix[H + 1];
    __local acc_t card_prefix[CS];
    __local acc_t lminus[52];
    __local acc_t part[WG];

    TermDesc d = descs[get_group_id(0)];
    int lid = get_local_id(0);
    int opp = player ^ 1;
    __global float* out = vec + (size_t)d.inst * H;
    __global const int2* player_cards = cards + player * H;
    __global const int2* opp_cards = cards + opp * H;

    if (d.kind == 0) {
        // someone folded: inclusion-exclusion over the opponent's hands that do not share a card
        __global const int* opp_off = card_off + opp * 53;
        acc_t s = 0;
        for (int j = lid; j < num_opp; j += WG) {
            int2 c = opp_cards[j];
            bool valid = c.x != d.turn && c.y != d.turn && c.x != d.river && c.y != d.river;
            float r = valid ? out[j] : 0.0f;
            lreach[j] = r;
            s += (acc_t)r;
        }
        part[lid] = s;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid < 52) {
            acc_t m = 0;
            for (int k = opp_off[lid]; k < opp_off[lid + 1]; k++)
                m += (acc_t)lreach[card_list[k]];
            lminus[lid] = m;
        }
        for (int stride = WG / 2; stride > 0; stride >>= 1) {
            if (lid < stride)
                part[lid] += part[lid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        acc_t total = part[0];
        acc_t payoff = d.folded != player ? WIN(d) : LOSE(d);
        for (int h = lid; h < num_hands; h += WG) {
            int2 c = player_cards[h];
            bool valid = c.x != d.turn && c.y != d.turn && c.x != d.river && c.y != d.river;
            float v = 0.0f;
            if (valid) {
                int si = same[player * H + h];
                acc_t reach_same = si >= 0 ? (acc_t)lreach[si] : 0;
                v = (float)(payoff * (total + reach_same - lminus[c.x] - lminus[c.y]));
            }
            out[h] = v;
        }
        return;
    }

    // showdown
    size_t player_base = ((size_t)d.board * 2 + player) * H;
    size_t opp_base = ((size_t)d.board * 2 + opp) * H;
    __global const ushort* opp_card_sorted = card_sorted + ((size_t)d.board * 2 + opp) * CS;
    __global const int* opp_card_start = card_start + (d.board * 2 + opp) * 53;
    for (int j = lid; j < num_opp; j += WG)
        lreach[j] = strength[opp_base + j] != 0 ? out[j] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // prefix sums of each card group, in ascending strength order like the CPU's `cfreach_minus`
    if (lid < 52) {
        int begin = opp_card_start[lid];
        int end = opp_card_start[lid + 1] - 1;
        acc_t run = 0;
        for (int k = begin; k < end; k++) {
            card_prefix[k] = run;
            run += (acc_t)lreach[opp_card_sorted[k]];
        }
        card_prefix[end] = run;
    }

    // prefix sums over all of the opponent's hands
    int cnt = count[d.board * 2 + opp];
    int chunk = (cnt + WG - 1) / WG;
    int begin = min(lid * chunk, cnt);
    int end = min(begin + chunk, cnt);
    acc_t s = 0;
    for (int k = begin; k < end; k++)
        s += (acc_t)lreach[sorted[opp_base + k]];
    part[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);

    // inclusive scan of the chunk sums (Hillis-Steele), then each chunk starts at the previous chunk's total
    for (int offset = 1; offset < WG; offset <<= 1) {
        acc_t add = lid >= offset ? part[lid - offset] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        part[lid] += add;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == WG - 1)
        prefix[cnt] = part[lid];

    acc_t run = lid > 0 ? part[lid - 1] : 0;
    for (int k = begin; k < end; k++) {
        prefix[k] = run;
        run += (acc_t)lreach[sorted[opp_base + k]];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    bool raked = d.tie != 0.0f;
    acc_t total = prefix[cnt];
    for (int h = lid; h < num_hands; h += WG) {
        float v = 0.0f;
        if (strength[player_base + h] != 0) {
            int2 c = player_cards[h];
            ushort4 pos = card_pos[player_base + h];
            int start1 = opp_card_start[c.x];
            int start2 = opp_card_start[c.y];
            acc_t all1 = card_prefix[opp_card_start[c.x + 1] - 1];
            acc_t all2 = card_prefix[opp_card_start[c.y + 1] - 1];
            acc_t below = prefix[lt[player_base + h]] - card_prefix[start1 + pos.x] - card_prefix[start2 + pos.z];
            acc_t at_most_all = prefix[le[player_base + h]];
            acc_t at_most1 = card_prefix[start1 + pos.y];
            acc_t at_most2 = card_prefix[start2 + pos.w];
            if (!raked) {
                // two passes on the CPU: the hands we beat, then the hands that beat us
                acc_t above = (total - at_most_all) - (all1 - at_most1) - (all2 - at_most2);
                v = (float)(WIN(d) * below);
                v += (float)(LOSE(d) * above);
            } else {
                // three passes on the CPU; the hand holding both of our cards is removed twice and added back
                acc_t total_unblocked = total - all1 - all2;
                acc_t at_most = at_most_all - at_most1 - at_most2;
                int si = same[player * H + h];
                acc_t reach_same = si >= 0 ? (acc_t)lreach[si] : 0;
                v = (float)(WIN(d) * below + TIE(d) * (at_most - below + reach_same) +
                            LOSE(d) * (total_unblocked - at_most));
            }
        }
        out[h] = v;
    }
}
