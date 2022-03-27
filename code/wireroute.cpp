/**
 * Parallel VLSI Wire Routing via OpenMP
 * Name 1(andrew_id 1), Name 2(andrew_id 2)
 */

#include "wireroute.h"
#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <algorithm>
#include <cmath>

static int _argc;
static const char **_argv;

const char *get_option_string(const char *option_name, const char *default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return _argv[i + 1];
    return default_value;
}

int get_option_int(const char *option_name, int default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return atoi(_argv[i + 1]);
    return default_value;
}

float get_option_float(const char *option_name, float default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return (float)atof(_argv[i + 1]);
    return default_value;
}

static void show_help(const char *program_path) {
    printf("Usage: %s OPTIONS\n", program_path);
    printf("\n");
    printf("OPTIONS:\n");
    printf("\t-f <input_filename> (required)\n");
    printf("\t-n <num_of_threads> (required)\n");
    printf("\t-p <SA_prob>\n");
    printf("\t-i <SA_iters>\n");
}

static void print_cost(int dim_x, int dim_y, cost_t* costs){
    printf("--------------PRINTING COST ARRAY----------------\n");
    printf("%d %d\n", dim_y, dim_x);

        for (int row = 0; row < dim_y; row++){
            for (int col = 0; col < dim_x; col++){
                printf( "%d ", costs[col + row * dim_x]);
            }
            printf("\n");
        }
    printf("---------------- DONE PRINTING ------------------\n");
}

// calculate the cost of path for one wire using the cost array
static int cost_calc(wire_t wire, cost_t *costs, int dim_x, int dim_y){
    //printf("ENTERING COST_CALCULATION......\n");

    int x_step, y_step;
    int total_cost = 0;

    if (wire.startx == wire.endx){
        if (wire.starty < wire.endy){
            for (y_step = wire.starty; y_step <= wire.endy; y_step++){
                total_cost += costs[wire.startx + dim_x * y_step];
            }
        } else{
            for (y_step = wire.starty; y_step >= wire.endy; y_step--){
                total_cost += costs[wire.startx + dim_x * y_step];
            }
        }
    } else if (wire.starty == wire.endy){
        if (wire.startx < wire.endx){
            for (x_step = wire.startx; x_step <= wire.endx; x_step++){
                total_cost += costs[x_step + dim_x * wire.endy];
            }
        } else{
            for (x_step = wire.startx; x_step >= wire.endx; x_step--){
                total_cost += costs[x_step + dim_x * wire.endy];
            }
        }
    }

    if (wire.bend_1){
        // start to bend 1
        if (wire.startx == wire.bend_1x){
            if (wire.starty < wire.bend_1y){
                for (y_step = wire.starty; y_step < wire.bend_1y; y_step++){
                    total_cost += costs[wire.startx + dim_x * y_step];
                }
            } else{
                for (y_step = wire.starty; y_step > wire.bend_1y; y_step--){
                    total_cost += costs[wire.startx + dim_x * y_step];
                }
            }
        } else if (wire.starty == wire.bend_1y){
            if (wire.startx < wire.bend_1x){
                for (x_step = wire.startx; x_step < wire.bend_1x; x_step++){
                    total_cost += costs[x_step + dim_x * wire.starty];
                }
            } else{
                for (x_step = wire.startx; x_step > wire.bend_1x; x_step--){
                    total_cost += costs[x_step + dim_x * wire.starty];
                }
            }
        }

        if (wire.bend_2){
            // bend 1 to bend 2
            if (wire.bend_2x == wire.bend_1x){
                // vertical bend 1 to bend 2
                if (wire.bend_1y < wire.bend_1y){
                    for (y_step = wire.starty; y_step < wire.bend_1y; y_step++){
                        total_cost += costs[wire.startx + dim_x * y_step];
                    }
                } else{
                    for (y_step = wire.starty; y_step > wire.bend_1y; y_step--){
                        total_cost += costs[wire.startx + dim_x * y_step];
                    }
                }
            } else if (wire.bend_2y == wire.bend_1y){
                if (wire.bend_1x < wire.bend_2x){
                    for (x_step = wire.bend_1x; x_step < wire.bend_2x; x_step++){
                        total_cost += costs[wire.startx + dim_x * y_step];
                    }
                } else{
                    for (x_step = wire.bend_1x; x_step > wire.bend_2x; x_step--){
                        total_cost += costs[wire.startx + dim_x * y_step];
                    }
                }
            }

            // bend 2 to end
            if (wire.endx == wire.bend_2x){
                // vertical from bend 2 to end
                if (wire.bend_2y < wire.endy){
                    for (y_step = wire.bend_2y; y_step <= wire.endy; y_step++){
                        total_cost += costs[wire.endx + dim_x * y_step];
                    }
                } else{
                    for (y_step = wire.bend_2y; y_step >= wire.endy; y_step--){
                        total_cost += costs[wire.endx + dim_x * y_step];
                    }
                }
            } else if (wire.endy == wire.bend_2y){
                // horizontal from bend 2 to end
                if (wire.bend_2x < wire.endx){
                    for (x_step = wire.bend_2x; x_step <= wire.endx; x_step++){
                        total_cost += costs[x_step + dim_x * wire.endy];
                    }
                } else{
                    for (x_step = wire.bend_2x; x_step >= wire.endx; x_step--){
                        total_cost += costs[x_step + dim_x * wire.endy];
                    }
                }
            }
        } else{
            // only 1 bend
            if (wire.endx == wire.bend_1x){
                // vertical from bend 1 to end
                if (wire.bend_1y < wire.endy){
                    for (y_step = wire.bend_1y; y_step <= wire.endy; y_step++){
                        total_cost += costs[wire.endx + dim_x * y_step];
                    }
                } else{
                    for (y_step = wire.bend_1y; y_step >= wire.endy; y_step--){
                        total_cost += costs[wire.endx + dim_x * y_step];
                    }
                }
            } else if (wire.endy == wire.bend_1y){
                // horizontal from bend 1 to end
                if (wire.bend_1x < wire.endx){
                    for (x_step = wire.bend_1x; x_step <= wire.endx; x_step++){
                        total_cost += costs[x_step + dim_x * wire.endy];
                    }
                } else{
                    for (x_step = wire.bend_1x; x_step >= wire.endx; x_step--){
                        total_cost += costs[x_step + dim_x * wire.endy];
                    }
                }
            }
        }
    }
    return total_cost;
}

// check to see if the start points and end points are on a straight line
static bool on_straight_line(wire_t wire){
    return (wire.startx == wire.endx || wire.starty == wire.endy);
}

static void add_cost(wire_t wire, cost_t *costs, int dim_x, int dim_y){
    //printf("ENTERING ADD_COST......\n");

    int y_step, x_step;

    if (wire.bend_1){
        if (wire.startx == wire.bend_1x){
            // vertical until bend 1
            if (wire.starty < wire.bend_1y){
                for (y_step = wire.starty; y_step < wire.bend_1y; y_step++){
                    costs[wire.startx + dim_x * y_step] += 1;
                }
            } else{
                for (y_step = wire.starty; y_step > wire.bend_1y; y_step--){
                    costs[wire.startx + dim_x * y_step] += 1;
                }
            }

            if (wire.bend_2){
                // horizontal until bend 2
                if (wire.bend_1x < wire.bend_2x){
                    for (x_step = wire.bend_1x; x_step < wire.bend_2x; x_step++){
                        costs[x_step + dim_x * wire.bend_1y] += 1;
                    }
                } else{
                    for (x_step = wire.bend_1x; x_step > wire.bend_2x; x_step--){
                        costs[x_step + dim_x * wire.bend_1y] += 1;
                    }
                }

                // vertical from bend 2 to end
                if (wire.bend_2y < wire.endy){
                    for (y_step = wire.bend_2y; y_step <= wire.endy; y_step++){
                        costs[wire.bend_2x + dim_x * y_step] += 1;
                    }
                } else {
                    for (y_step = wire.bend_2y; y_step >= wire.endy; y_step--){
                        costs[wire.bend_2x + dim_x * y_step] += 1;
                    }
                }
            } else{
                // only bend 1

                // horizontal from bend 1 to end
                if (wire.bend_1x < wire.endx){
                    for (x_step = wire.bend_1x; x_step <= wire.endx; x_step++){
                        costs[x_step + dim_x * wire.bend_1y] += 1;
                    }
                } else{
                    for (x_step = wire.bend_1x; x_step >= wire.endx; x_step--){
                        costs[x_step + dim_x * wire.bend_1y] += 1;
                    }
                }
            }

        } else if (wire.starty == wire.bend_1y){
            // horizontal from start to bend 1

            if (wire.startx < wire.bend_1x){
                for (x_step = wire.startx; x_step < wire.bend_1x; x_step++){
                    costs[x_step + dim_x * wire.starty] += 1;
                }
            } else{
                for (x_step = wire.startx; x_step > wire.bend_1x; x_step--){
                    costs[x_step + dim_x * wire.starty] += 1;
                }
            }

            if (wire.bend_2){
                // vertical until bend 2
                if (wire.bend_1y < wire.bend_2y){
                    for (y_step = wire.bend_1y; y_step < wire.bend_2y; y_step++){
                        costs[wire.bend_1x + dim_x * y_step] += 1;
                    }
                } else{
                    for (y_step = wire.bend_1y; y_step > wire.bend_2y; y_step--){
                        costs[wire.bend_1x + dim_x * y_step] += 1;
                    }
                }

                // horizontal from bend 2 to end
                if (wire.bend_2x < wire.endx){
                    for (x_step = wire.bend_2x; x_step <= wire.endx; x_step++){
                        costs[x_step + dim_x * wire.bend_2y] += 1;
                    }
                } else {
                    for (x_step = wire.bend_2x; x_step >= wire.endx; x_step--){
                        costs[x_step + dim_x * wire.bend_2y] += 1;
                    }
                }
            } else{
                // only bend 1

                // vertical from bend 1 to end
                if (wire.bend_1y < wire.endy){
                    for (y_step = wire.bend_1y; y_step <= wire.endy; y_step++){
                        costs[wire.bend_1x + dim_x * y_step] += 1;
                        // printf("index: (%d, %d), cost: %d\n", wire.bend_1x, step, 
                        //         costs[wire.bend_1x + dim_x * step]);
                    }
                } else{
                    for (y_step = wire.bend_1y; y_step >= wire.endy; y_step--){
                        costs[wire.bend_1x + dim_x * y_step] += 1;
                        // printf("index: (%d, %d), cost: %d\n", wire.bend_1x, step, 
                        //         costs[wire.bend_1x + dim_x * step]);
                    }
                }
            }
        } 
    } else{
        // no bends
        if (wire.startx == wire.endx){
            // vertical line from start to end
            if (wire.starty < wire.endy){
                for (y_step = wire.starty; y_step <= wire.endy; y_step++){
                    costs[wire.startx + dim_x * y_step] += 1;
                }
            } else{
                for (y_step = wire.starty; y_step >= wire.endy; y_step--){
                    costs[wire.startx + dim_x * y_step] += 1;
                }
            }
            //printf("wire #%d with cost %d\n", i, wires[i].total_cost);
        } else if (wire.starty == wire.endy){
            if (wire.startx < wire.endx){
                for (x_step = wire.startx; x_step <= wire.endx; x_step++){
                    costs[x_step + dim_x * wire.starty] += 1;
                }
            } else{
                for (x_step = wire.startx; x_step >= wire.endx; x_step--){
                    costs[x_step + dim_x * wire.starty] += 1;
                }
            }
        }
    }
}

// clear the costs in the cost array along the existing route
static void clear_cost(wire_t wire, cost_t *costs, int dim_x, int dim_y){
    //printf("ENTERING CLEAR_COST......\n");

    int x_step, y_step;

    if (wire.bend_1){
        if (wire.startx == wire.bend_1x){
            // vertical from start to bend 1
            if (wire.starty < wire.bend_1y){
                for (y_step = wire.starty; y_step < wire.bend_1y; y_step++){
                    costs[wire.startx + dim_x * y_step] -= 1;
                }
            } else{
                for (y_step = wire.starty; y_step > wire.bend_1y; y_step--){
                    costs[wire.startx + dim_x * y_step] -= 1;
                }
            }

            if (wire.bend_2){
                // horizontal until bend 2
                if (wire.bend_1x < wire.bend_2x){
                    for (x_step = wire.bend_1x; x_step < wire.bend_2x; x_step++){
                        //printf("index in cost arr: %d\n", step + dim_y * wires[i].starty);
                        costs[x_step + dim_x * wire.bend_1y] -= 1;
                    }
                } else{
                    for (x_step = wire.bend_1x; x_step > wire.bend_2x; x_step--){
                        //printf("index in cost arr: %d\n", step + dim_y * wires[i].starty);
                        costs[x_step + dim_x * wire.bend_1y] -= 1;
                    }
                }

                // vertical from bend 2 to end
                if (wire.bend_2y < wire.endy){
                    for (y_step = wire.bend_2y; y_step <= wire.endy; y_step++){
                        costs[wire.bend_2x + dim_x * y_step] -= 1;
                    }
                } else {
                    for (y_step = wire.bend_2y; y_step >= wire.endy; y_step--){
                        costs[wire.bend_2x + dim_x * y_step] -= 1;
                    }
                }
            } else{
                // only bend 1

                // horizontal from bend 1 to end
                if (wire.bend_1x < wire.endx){
                    for (x_step = wire.bend_1x; x_step <= wire.endx; x_step++){
                        //printf("index in cost arr: %d\n", step + dim_y * wires[i].starty);
                        costs[x_step + dim_x * wire.bend_1y] -= 1;
                    }
                } else{
                    for (x_step = wire.bend_1x; x_step >= wire.endx; x_step--){
                        //printf("index in cost arr: %d\n", step + dim_y * wires[i].starty);
                        costs[x_step + dim_x * wire.bend_1y] -= 1;
                    }
                }
            }

        } else if (wire.starty == wire.bend_1y){
            // horizontal from start to bend 1
            if (wire.startx < wire.bend_1x){
                for (x_step = wire.startx; x_step < wire.bend_1x; x_step++){
                    //printf("index in cost arr: %d\n", step + dim_y * wires[i].starty);
                    costs[x_step + dim_x * wire.starty] -= 1;
                }
            } else{
                for (x_step = wire.startx; x_step > wire.bend_1x; x_step--){
                    //printf("index in cost arr: %d\n", step + dim_y * wires[i].starty);
                    costs[x_step + dim_x * wire.starty] -= 1;
                }
            }

            if (wire.bend_2){
                // vertical from bend 1 to bend 2
                if (wire.bend_1y < wire.bend_2y){
                    for (y_step = wire.bend_1y; y_step < wire.bend_2y; y_step++){
                        costs[wire.bend_1x + dim_x * y_step] -= 1;
                    }
                } else{
                    for (y_step = wire.bend_1y; y_step > wire.bend_2y; y_step--){
                        costs[wire.bend_1x + dim_x * y_step] -= 1;
                    }
                }

                // horizontal from bend 2 to end
                if (wire.bend_2x < wire.endx){
                    for (x_step = wire.bend_2x; x_step <= wire.endx; x_step++){
                        costs[x_step + dim_x * wire.bend_2y] -= 1;
                    }
                } else {
                    for (x_step = wire.bend_2x; x_step >= wire.endx; x_step--){
                        costs[x_step + dim_x * wire.bend_2y] -= 1;
                    }
                }
            } else{
                // only bend 1

                // vertical from bend 1 to end
                if (wire.bend_1y < wire.endy){
                    for (y_step = wire.bend_1y; y_step <= wire.endy; y_step++){
                        costs[wire.bend_1x + dim_x * y_step] -= 1;
                    }
                } else{
                    for (y_step = wire.bend_1y; y_step >= wire.endy; y_step--){
                        costs[wire.bend_1x + dim_x * y_step] -= 1;
                    }
                }
            }
        } 
    } else{
        // no bends
        if (wire.startx == wire.endx){
            // vertical line from start to end
            if (wire.starty < wire.endy){
                for (y_step = wire.starty; y_step <= wire.endy; y_step++){
                    costs[wire.startx + dim_x * y_step] -= 1;
                }
            } else{
                for (y_step = wire.starty; y_step >= wire.endy; y_step--){
                    costs[wire.startx + dim_x * y_step] -= 1;
                }
            }

            //printf("wire #%d with cost %d\n", i, wires[i].total_cost);
        } else if (wire.starty == wire.endy){
            if (wire.startx < wire.endx){
                for (x_step = wire.startx; x_step <= wire.endx; x_step++){
                    costs[x_step + dim_x * wire.starty] -= 1;
                }
            } else{
                for (x_step = wire.startx; x_step >= wire.endx; x_step--){
                    costs[x_step + dim_x * wire.starty] -= 1;
                }
            }
        }
    }
}

// perform the wire routing iterations (sequential for now)
static void routing(wire_t *wires, cost_t *costs, int dim_x, int dim_y, 
                    int num_wires, int N, int num_threads){
    // loop iterations for improvement (inside which each wire is checked)
    //printf("ENTERING ROUTING...\n");
    for (int i = 0; i < N; i++){
        // loop each wire

        // PARALLELIZE cross wires (num_wires / num_threads = wires taken care by one thread)
        // SHARED: costs, wires

        int WIRES_PER_THREAD = (num_wires + num_threads - 1) / num_threads;
        omp_set_num_threads(num_threads);
        int wid;
        #pragma omp parallel for schedule(static, WIRES_PER_THREAD) shared(costs, wires)
        for (wid = 0; wid < num_wires; wid++){

            int ori_cost;
            #pragma omp critical
            {
                ori_cost = cost_calc(wires[wid], costs, dim_x, dim_y);
            }
            // printf("LOOP THREAD #%d\n", wid);
            // printf("original cost: %d\n", ori_cost);

            wire_t cur_wire = wires[wid];
            int route_len = 1+abs(cur_wire.endx - cur_wire.startx) + abs(cur_wire.endy - cur_wire.starty);
            int min_cost = ori_cost;
            wire_t best_route;

            // clear the current costs, updates costs, needs synchronize
            #pragma omp critical
            {   
                clear_cost(wires[wid], costs, dim_x, dim_y);
            }

            // calculate the current path cost, set to minimum
            best_route = cur_wire;

            int x_step, y_step;
            int cur_cost;

            // calculate costs of other alternatives, compare
            // skipped for straight line
            if (!on_straight_line(cur_wire)){

                // horizontal travel first
                if (cur_wire.startx < cur_wire.endx){
                    // travel horizontally does not include itself
                    // loops for all horizontal possibilities
                    // PARALLELIZE calculating all possible routes
                    // SHARED: min_cost, best_route (specific to that wire)
                    for (x_step = cur_wire.startx; x_step < cur_wire.endx; x_step++){
                        wire_t new_w;

                        new_w.startx = cur_wire.startx;
                        new_w.starty = cur_wire.starty;

                        new_w.bend_1 = true;
                        new_w.bend_1x = x_step+1;
                        new_w.bend_1y = cur_wire.starty;

                        new_w.endx = cur_wire.endx;
                        new_w.endy = cur_wire.endy;
                        new_w.total_cost = 0;

                        //check if second bend is needed
                        // second bend would be vertical line
                        if (new_w.bend_1x != new_w.endx){
                            // first bend is not on same line as end points
                            new_w.bend_2 = true;

                            //second bend has to have the same y value as the end point
                            new_w.bend_2y = new_w.endy;
                            new_w.bend_2x = new_w.bend_1x;
                        } else{
                            new_w.bend_2 = false;
                        }

                        int cur_cost;
                        #pragma omp critical
                        {
                            //print_cost(dim_x, dim_y, costs);
                            cur_cost = cost_calc(new_w, costs, dim_x, dim_y) + route_len;
                            // printf("travel horizontally first cost: %d\n", cur_cost);
                            // printf("bend 1: (%d, %d)\n", new_w.bend_1x, new_w.bend_1y);
                            // if (new_w.bend_2){
                            //     printf("bend 2: (%d, %d)\n", new_w.bend_2x, new_w.bend_2y);
                            // }
                            if (cur_cost < min_cost){
                                min_cost = cur_cost;
                                best_route = new_w;
                            }
                        }
                    }
                } else{
                    for (x_step = cur_wire.startx; x_step > cur_wire.endx; x_step--){
                        wire_t new_w;

                        new_w.startx = cur_wire.startx;
                        new_w.starty = cur_wire.starty;

                        new_w.bend_1 = true;
                        new_w.bend_1x = x_step-1;
                        new_w.bend_1y = cur_wire.starty;

                        new_w.endx = cur_wire.endx;
                        new_w.endy = cur_wire.endy;
                        new_w.total_cost = 0;

                        //check if second bend is needed
                        // second bend would be vertical line
                        if (new_w.bend_1x != new_w.endx){
                            // first bend is not on same line as end points
                            new_w.bend_2 = true;

                            //second bend has to have the same y value as the end point
                            new_w.bend_2y = new_w.endy;
                            new_w.bend_2x = new_w.bend_1x;
                        } else{
                            new_w.bend_2 = false;
                        }

                        #pragma omp critical
                        {   
                            //print_cost(dim_x, dim_y, costs);
                            cur_cost = cost_calc(new_w, costs, dim_x, dim_y);
                            // printf("travel horizontally first cost: %d\n", cur_cost) + route_len;
                            // printf("bend 1: (%d, %d)\n", new_w.bend_1x, new_w.bend_1y);
                            // printf("bend 2: (%d, %d)\n", new_w.bend_2x, new_w.bend_2y);
                            if (cur_cost < min_cost){
                                min_cost = cur_cost;
                                best_route = new_w;
                            }
                        }
                    }
                }

                // horizontal travel first

                if (cur_wire.starty < cur_wire.endy){
                    // travel horizontally does not include itself
                    // loop for all possibilities traveling vertically
                    for (y_step = cur_wire.starty; y_step < cur_wire.endy; y_step++){

                        wire_t new_w;
                        new_w.startx = cur_wire.startx;
                        new_w.starty = cur_wire.starty;

                        new_w.bend_1 = true;
                        new_w.bend_1x = cur_wire.startx;
                        new_w.bend_1y = y_step+1;

                        new_w.endx = cur_wire.endx;
                        new_w.endy = cur_wire.endy;
                        new_w.total_cost = 0;

                        //check if second bend is needed
                        // second bend would be vertical line
                        if (new_w.bend_1x != new_w.endx){
                            // first bend is not on same line as end points
                            new_w.bend_2 = true;

                            //second bend has to have the same y value as the end point
                            new_w.bend_2y = new_w.bend_1y;
                            new_w.bend_2x = new_w.endx;
                        } else{
                            new_w.bend_2 = false;
                        }

                        #pragma omp critical
                        {
                            //print_cost(dim_x, dim_y, costs);
                            cur_cost = cost_calc(new_w, costs, dim_x, dim_y);
                            // printf("travel vertically first cost: %d\n", cur_cost) + route_len;
                            // printf("bend 1: (%d, %d)\n", new_w.bend_1x, new_w.bend_1y);
                            // printf("bend 2: (%d, %d)\n", new_w.bend_2x, new_w.bend_2y);
                            if (cur_cost < min_cost){
                                min_cost = cur_cost;
                                best_route = new_w;
                            }
                        } 
                    }
                } else{
                    for (y_step = cur_wire.starty; y_step > cur_wire.endy; y_step--){

                        wire_t new_w;
                        new_w.startx = cur_wire.startx;
                        new_w.starty = cur_wire.starty;

                        new_w.bend_1 = true;
                        new_w.bend_1x = cur_wire.startx;
                        new_w.bend_1y = y_step-1;

                        new_w.endx = cur_wire.endx;
                        new_w.endy = cur_wire.endy;
                        new_w.total_cost = 0;

                        //check if second bend is needed
                        // second bend would be vertical line
                        if (new_w.bend_1x != new_w.endx){
                            // first bend is not on same line as end points
                            new_w.bend_2 = true;

                            //second bend has to have the same y value as the end point
                            new_w.bend_2y = new_w.bend_1y;
                            new_w.bend_2x = new_w.endx;
                        } else{
                            new_w.bend_2 = false;
                        }

                        #pragma omp critical
                        {
                            //print_cost(dim_x, dim_y, costs);
                            cur_cost = cost_calc(new_w, costs, dim_x, dim_y) + route_len;
                            // printf("travel vertically first cost: %d\n", cur_cost);
                            // printf("bend 1: (%d, %d)\n", new_w.bend_1x, new_w.bend_1y);
                            // printf("bend 2: (%d, %d)\n", new_w.bend_2x, new_w.bend_2y);
                            if (cur_cost < min_cost){
                                min_cost = cur_cost;
                                best_route = new_w;
                            } 
                        }
                    }
                }

            // update the wire contents and costs
            // CRITICAL SECTION
            //#pragma omp critical
            //{
                // update the route, needs synchronize
                // add the cost of new route, updates costs, needs synchronize
                
                //}
            //} 
            }

            #pragma omp critical
            {
                wires[wid] = best_route;
                add_cost(wires[wid], costs, dim_x, dim_y);
                // printf("BEST ROUTE FOUND: \n");
                // printf("BEST bend 1: (%d, %d)\n", wires[wid].bend_1x, wires[wid].bend_1y);
                // printf("BEST bend 2: (%d, %d)\n", wires[wid].bend_2x, wires[wid].bend_2y);
                // print_cost(dim_x, dim_y, costs);
                // printf("EXISTING THIS THREAD");
            }
        }
    }
}

int main(int argc, const char *argv[]) {
    using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double> dsec;

    auto init_start = Clock::now();
    double init_time = 0;

    _argc = argc - 1;
    _argv = argv + 1;

    const char *input_filename = get_option_string("-f", NULL);
    int num_of_threads = get_option_int("-n", 1);
    double SA_prob = get_option_float("-p", 0.1f);
    int SA_iters = get_option_int("-i", 5);

    int error = 0;

    if (input_filename == NULL) {
        printf("Error: You need to specify -f.\n");
        error = 1;
    }

    if (error) {
        show_help(argv[0]);
        return 1;
    }

    printf("Number of threads: %d\n", num_of_threads);
    printf("Probability parameter for simulated annealing: %lf.\n", SA_prob);
    printf("Number of simulated annealing iterations: %d\n", SA_iters);
    printf("Input file: %s\n", input_filename);

    FILE *input = fopen(input_filename, "r");

    if (!input) {
        printf("Unable to open file: %s.\n", input_filename);
        return 1;
    }

    int dim_x, dim_y;
    int num_of_wires;

    fscanf(input, "%d %d\n", &dim_y, &dim_x);
    fscanf(input, "%d\n", &num_of_wires);

    wire_t *wires = (wire_t *)calloc(num_of_wires, sizeof(wire_t));
    /* Read the grid dimension and wire information from file */
    printf("about to enter loop for wires......\n");
    for (int widx = 0; widx < num_of_wires; widx++){
        int cur_startx, cur_starty, cur_endx, cur_endy;
        fscanf(input, "%d %d %d %d\n", &cur_startx, &cur_starty, &cur_endx, &cur_endy);
        // printf("start x: %d. start y: %d \n", cur_startx, cur_starty);
        // printf("end x: %d. end y: %d \n", cur_endx, cur_endy);

        wires[widx].startx = cur_startx;
        wires[widx].starty = cur_starty;
        wires[widx].endx = cur_endx;
        wires[widx].endy = cur_endy;

        wires[widx].bend_1 = false;
        wires[widx].bend_2 = false;

        wires[widx].total_cost = 0;
        //printf("start x: %d. start y: %d \n", wires[widx].startx, wires[widx].starty);
    }

    cost_t *costs = (cost_t *)calloc(dim_x * dim_y, sizeof(cost_t));
    /* Initialize cost matrix */
    for (int i = 0; i < dim_x * dim_y; i++){
        costs[i] = 0;
    }

    printf("about to enter loop for initialization......\n");
    /* Initailize additional data structures needed in the algorithm */
    for (int i = 0; i < num_of_wires; i++){
        //printf("new wire #%d\n", i);
        if (wires[i].startx == wires[i].endx){
            
            int y_min = std::min(wires[i].starty, wires[i].endy);
            int y_max = std::max(wires[i].starty, wires[i].endy);

            for (int step = y_min; step <= y_max; step++){
                //printf("index in cost arr: %d\n", wires[i].startx + dim_y * step);
                costs[wires[i].startx + dim_x * step] += 1;
                // printf("index: (%d, %d), cost: %d\n", wires[i].startx, step, 
                //         costs[wires[i].startx + dim_y * step]);
            }

            //printf("wire #%d with cost %d\n", i, wires[i].total_cost);
        } else if (wires[i].starty == wires[i].endy){
            int x_min = std::min(wires[i].startx, wires[i].endx);
            int x_max = std::max(wires[i].startx, wires[i].endx);

            for (int step = x_min; step <= x_max; step++){
                //printf("index in cost arr: %d\n", step + dim_y * wires[i].starty);
                costs[step + dim_x * wires[i].starty] += 1;
                // printf("index: (%d, %d), cost: %d\n", step, wires[i].starty, 
                //         costs[step + dim_y * wires[i].starty]);
            }

            //printf("wire #%d with cost %d\n", i, wires[i].total_cost);
        } else{
            // needs to bend somewhere
            wires[i].bend_1 = true;
            wires[i].bend_1x = wires[i].endx;
            wires[i].bend_1y = wires[i].starty;

            int x_min = std::min(wires[i].startx, wires[i].bend_1x);
            int x_max = std::max(wires[i].startx, wires[i].bend_1x);

            // horizontal line, x is changing
            for (int step = x_min; step <= x_max; step++){
                costs[step + dim_x * wires[i].starty] += 1;
                // printf("index: (%d, %d), cost: %d\n", step, wires[i].starty,
                //         costs[step + dim_x * wires[i].starty]);
            }

            int y_min, y_max;
            if (wires[i].bend_1y < wires[i].endy){
                y_min = wires[i].bend_1y + 1;
                y_max = wires[i].endy;
            } else{
                y_max = wires[i].bend_1y - 1;
                y_min = wires[i].endy;
            }

            // vertical line, y is changing
            for (int step = y_min; step <= y_max; step++){
                costs[wires[i].bend_1x + dim_x * step] += 1;
                // printf("index: (%d, %d), cost: %d\n", wires[i].bend_1x, step,
                //         costs[wires[i].bend_1x + dim_x * step]);
            }
        }
        
    }

    /* Conduct initial wire placement */
    

    init_time += duration_cast<dsec>(Clock::now() - init_start).count();
    printf("Initialization Time: %lf.\n", init_time);

    auto compute_start = Clock::now();
    double compute_time = 0;

    /**
     * Implement the wire routing algorithm here
     * Feel free to structure the algorithm into different functions
     * Don't use global variables.
     * Use OpenMP to parallelize the algorithm.
     */
    int N = 5;
    routing(wires, costs, dim_x, dim_y, num_of_wires, N, num_of_threads);
    // printf("ROUTING DONE!!!");
    // print_cost(dim_x, dim_y, costs);

    compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
    printf("Computation Time: %lf.\n", compute_time);

    /* Write wires and costs to files */
    char cost_filename[256];
    int n = sprintf(cost_filename, "cost_%s_%d", input_filename, num_of_threads);
    printf("cost_%s_%d", input_filename, num_of_threads);

    FILE *cost_output = fopen(cost_filename, "w+");

    if (cost_output != NULL){
        fprintf(cost_output, "%d %d\n", dim_y, dim_x);

        for (int row = 0; row < dim_y; row++){
            for (int col = 0; col < dim_x; col++){
                fprintf(cost_output, "%d ", costs[col + row * dim_x]);
            }
            fprintf(cost_output, "\n");
        }

        fclose(cost_output);
    }

    // wire content
    char wire_filename[256];
    n = sprintf(wire_filename, "output_%s_%d", input_filename, num_of_threads);
    printf(wire_filename);
    FILE *wire_output = fopen(wire_filename, "w+");

    wire_t wire;
    if (wire_output != NULL){
        fprintf(wire_output, "%d %d\n", dim_y, dim_x);
        fprintf(wire_output, "%d \n", num_of_wires);

        for (int w = 0; w < num_of_wires; w++){
            wire = wires[w];

            if (wire.bend_1){
                if (wire.startx == wire.bend_1x){
                    // vertical until bend 1
                    if (wire.starty < wire.bend_1y){
                        for (int y_step = wire.starty; y_step < wire.bend_1y; y_step++){
                            fprintf(wire_output, "%d %d ", wire.startx, y_step);
                        }
                    } else{
                        for (int y_step = wire.starty; y_step > wire.bend_1y; y_step--){
                            fprintf(wire_output, "%d %d ", wire.startx, y_step);
                        }
                    }

                    if (wire.bend_2){
                        // horizontal until bend 2
                        if (wire.bend_1x < wire.bend_2x){
                            for (int step = wire.bend_1x; step < wire.bend_2x; step++){
                                fprintf(wire_output, "%d %d ", step, wire.bend_1y);
                            }
                        } else{
                            for (int step = wire.bend_1x; step > wire.bend_2x; step--){
                                fprintf(wire_output, "%d %d ", step, wire.bend_1y);
                            }
                        }

                        // vertical from bend 2 to end
                        if (wire.bend_2y < wire.endy){
                            for (int y_step = wire.bend_2y; y_step <= wire.endy; y_step++){
                                fprintf(wire_output, "%d %d ", wire.bend_2x, y_step);
                            }
                        } else {
                            for (int y_step = wire.bend_2y; y_step >= wire.endy; y_step--){
                                fprintf(wire_output, "%d %d ", wire.bend_2x, y_step);
                            }
                        }

                    } else{
                        // only bend 1

                        // horizontal from bend 1 to end
                        if (wire.bend_1x < wire.endx){
                            for (int step = wire.bend_1x; step <= wire.endx; step++){
                                fprintf(wire_output, "%d %d ", step, wire.bend_1y);
                            }
                        } else{
                            for (int step = wire.bend_1x; step >= wire.endx; step--){
                                fprintf(wire_output, "%d %d ", step, wire.bend_1y);
                            }
                        }
                    }

                } else if (wire.starty == wire.bend_1y){
                    // horizontal from start to bend 1
                    if (wire.startx < wire.bend_1x){
                        for (int step = wire.startx; step < wire.bend_1x; step++){
                            fprintf(wire_output, "%d %d ", step, wire.starty);
                        }
                    } else{
                        for (int step = wire.startx; step > wire.bend_1x; step--){
                            fprintf(wire_output, "%d %d ", step, wire.starty);
                        }
                    }

                    if (wire.bend_2){
                        // vertical from bend 1 to bend 2
                        if (wire.bend_1y < wire.bend_2y){
                            for (int step = wire.bend_1y; step < wire.bend_2y; step++){
                                fprintf(wire_output, "%d %d ", wire.bend_1x, step);
                            }
                        } else{
                            for (int step = wire.bend_1y; step > wire.bend_2y; step--){
                                fprintf(wire_output, "%d %d ", wire.bend_1x, step);
                            }
                        }

                        // horizontal from bend 2 to end
                        if (wire.bend_2x < wire.endx){
                            for (int x_step = wire.bend_2x; x_step <= wire.endx; x_step++){
                                fprintf(wire_output, "%d %d ", x_step, wire.bend_2y);
                            }
                        } else {
                            for (int x_step = wire.bend_2x; x_step >= wire.endx; x_step--){
                                fprintf(wire_output, "%d %d ", x_step, wire.bend_2y);
                            }
                        }

                    } else{
                        // only bend 1

                        // vertical from bend 1 to end
                        if (wire.bend_1y < wire.endy){
                            for (int step = wire.bend_1y; step <= wire.endy; step++){
                                fprintf(wire_output, "%d %d ", wire.endx, step);
                            }
                        } else{
                            for (int step = wire.bend_1y; step >= wire.endy; step--){
                                fprintf(wire_output, "%d %d ", wire.endx, step);
                            }
                        }
                    }
                } 
            } else{
                // no bends
                if (wire.startx == wire.endx){
                    // vertical line from start to end
                    if (wire.starty < wire.endy){
                        for (int step = wire.starty; step <= wire.endy; step++){
                            fprintf(wire_output, "%d %d ", wire.startx, step);
                        }
                    } else{
                        for (int step = wire.starty; step >= wire.endy; step--){
                            fprintf(wire_output, "%d %d ", wire.startx, step);
                        }
                    }

                } else if (wire.starty == wire.endy){
                    if (wire.startx < wire.endx){
                        for (int step = wire.startx; step <= wire.endx; step++){
                            fprintf(wire_output, "%d %d ", step, wire.endy);
                        }
                    } else{
                        for (int step = wire.startx; step >= wire.endx; step--){
                            fprintf(wire_output, "%d %d ", step, wire.endy);
                        }
                    }
                }
            }
            // fprintf(wire_output, "%d %d ", wires[w].bend_1x, wires[w].bend_1y);

            fprintf(wire_output, "\n");
        }

        fclose(wire_output);
    }

    //printf("owari\n");
    return 0;
}
