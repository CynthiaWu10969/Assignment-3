/**
 * Parallel VLSI Wire Routing via OpenMP
 * Name 1(andrew_id 1), Name 2(andrew_id 2)
 */

#ifndef __WIREOPT_H__
#define __WIREOPT_H__

#include <omp.h>

typedef struct { /* Define the data structure for wire here */
    int startx;
    int starty;
    int endx;
    int endy;

    bool bend_1;
    bool bend_2;

    int bend_1x;
    int bend_1y;
    int bend_2x;
    int bend_2y;

    int total_cost;
} wire_t;

typedef int cost_t;

const char *get_option_string(const char *option_name, const char *default_value);
int get_option_int(const char *option_name, int default_value);
float get_option_float(const char *option_name, float default_value);

#endif
