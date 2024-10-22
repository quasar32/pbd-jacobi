#define FPS 60
#define N_BEADS 8 

#define DT (1.0f / FPS)
#define N_STEPS 100
#define SDT (DT / N_STEPS)
#define STS (N_STEPS * FPS)

struct wire {
    float2 pos;
    float radius;
};

struct group {
    struct wire wire;
    float beads_radius[N_BEADS];
    float beads_mass[N_BEADS];
    float2 beads_pos[N_BEADS];
    float2 beads_prev_pos[N_BEADS];
    float2 beads_next_pos[N_BEADS];
    float2 beads_vel[N_BEADS];
};

__constant float2 gravity = {0.0F, -10.0F};

__kernel void start_step(__global struct group *groups) {
    int i = get_global_id(0);
    __global struct group *g = &groups[i / N_BEADS]; 
    i %= N_BEADS;
    g->beads_vel[i] += gravity * SDT; 
    g->beads_prev_pos[i] = g->beads_pos[i]; 
    g->beads_pos[i] += g->beads_vel[i] * SDT; 
}

__kernel void end_step(__global struct group *groups) {
    int i = get_global_id(0);
    __global struct group *g = &groups[i / N_BEADS]; 
    i %= N_BEADS;
    g->beads_pos[i] = g->beads_next_pos[i];
    g->beads_vel[i] = g->beads_pos[i] - g->beads_prev_pos[i];
    g->beads_vel[i] = g->beads_vel[i] * STS;
}

void bead_col(__global struct group *g, int i, int j) {
    float2 dir = g->beads_pos[j] - g->beads_pos[i];
    float d = length(dir);
    float ra = g->beads_radius[i];
    float rb = g->beads_radius[j];
    if (d == 0.0f || d > ra + rb)
        return;
    dir = dir / d;
    float corr = (ra + rb - d) / 2.0f;
    g->beads_next_pos[i] = g->beads_pos[i] - dir * corr;
    g->beads_next_pos[j] = g->beads_pos[j] + dir * corr;
    float v0a = dot(g->beads_vel[i], dir); 
    float v0b = dot(g->beads_vel[j], dir); 
    float ma = g->beads_mass[i];
    float mb = g->beads_mass[j];
    float mt = ma + mb;
    float vc = ma * v0a + mb * v0b;
    float v1a = (vc - mb * (v0a - v0b)) / mt;
    float v1b = (vc - ma * (v0b - v0a)) / mt; 
    g->beads_vel[i] = g->beads_vel[i] + dir * (v1a - v0a); 
    g->beads_vel[j] = g->beads_vel[j] + dir * (v1b - v0b); 
}

__kernel void keep_on_wire(__global struct group *groups) {
    int i = get_global_id(0);
    __global struct group *g = &groups[i / N_BEADS]; 
    i %= N_BEADS;
    float2 dir = g->beads_pos[i] - g->wire.pos;
    float len = length(dir); 
    if (len == 0.0f)
        return;
    dir = dir / len;
    float lambda = g->wire.radius - len;
    g->beads_next_pos[i] = g->beads_pos[i] + dir * lambda;
}

__kernel void bead_cols(__global struct group *groups) { 
    int i = get_global_id(0);
    __global struct group *g = &groups[i / N_BEADS]; 
    i %= N_BEADS;
    for (int j = 0; j < i; j++)
        bead_col(g, i, j);
}
