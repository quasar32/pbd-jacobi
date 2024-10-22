#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <CL/cl.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>
#include <string.h>

#define N_BEADS 8 
#define FPS 60

typedef struct __attribute__((aligned(8))) float2 {
  float x;
  float y;
} float2;

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

static struct group *groups;

static cl_ulong elapsed;
static cl_kernel start_step;
static cl_kernel keep_on_wire;
static cl_kernel end_step;
static cl_kernel bead_cols;
static cl_command_queue cmdq;
static cl_mem groups_mem;
static int n_groups = 1;
static int ends_only;
static FILE **csvs;

static void die(const char *fn, int err) {
  fprintf(stderr, "%s(%d)\n", fn, err);
  exit(EXIT_FAILURE);
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "eg:")) != -1) {
    switch (c) {
    case 'e':
      /* output only first and last frame */
      ends_only = 1;
      break;
    case 'g':
      /* number of groups*/
      if (sscanf(optarg, "%d\n", &n_groups) != 1) {
        fprintf(stderr, "group number is invalid\n");
        exit(1);
      }
      break;
    case '?': 
      exit(1);
    }
  }
}

static void *xmalloc(size_t sz) {
  void *ret = malloc(sz);
  if (!sz)
    die("malloc", errno);
  return ret;
}

static char *read_all(const char *path) {
  int fd = open(path, O_RDONLY);
  if (fd < 0)
    die("open", errno);
  struct stat st;
  if (fstat(fd, &st) < 0)
    die("stat", errno);
  char *str = xmalloc(st.st_size + 1);
  if (read(fd, str, st.st_size) < 0)
    die("read", errno);
  close(fd);
  str[st.st_size] = '\0';
  return str;
}

static void notify(const char *err, const void *info, size_t sz, void *user) {
    fprintf(stderr, "opencl-err: %s\n", err);
}

static cl_kernel create_kernel(const char *name, 
    cl_program prog, cl_device_id did) {
  int err;
  cl_kernel kernel = clCreateKernel(prog, name, &err);
  if (err != CL_SUCCESS)
    die("clCreateKernel", err);
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &groups_mem);
  if (err != CL_SUCCESS)
    die("clSetKernelArg", err);
  return kernel;
}

static void init_cl(void) {
  int err;
  cl_platform_id pid;
  err = clGetPlatformIDs(1, &pid, NULL);
  if (err != CL_SUCCESS)
    die("clGetPlatformIDs", err);
  cl_device_id did;
  err = clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 1, &did, NULL);
  if (err != CL_SUCCESS)
    die("clGetDeviceIDs", err);
  cl_context ctx = clCreateContext(NULL, 1, &did, notify, NULL, &err); 
  if (!ctx)
    die("clCreateContext", err);
  groups_mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | 
      CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY, 
      n_groups * sizeof(*groups), groups, &err);
  if (err != CL_SUCCESS)
    die("clCreateBuffer", err);
  const char *str = read_all("pbd.cl");
  cl_program prog = clCreateProgramWithSource(ctx, 1, &str, NULL, &err);
  free((void *) str);
  if (err != CL_SUCCESS)
    die("clCreateProgramWithSource", err);
  err = clBuildProgram(prog, 1, &did, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_sz;
    clGetProgramBuildInfo(prog, did, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
    char *log = xmalloc(log_sz);
    err = clGetProgramBuildInfo(prog, did, 
        CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
    if (err == CL_SUCCESS)
      fprintf(stderr, "cl-log: %s\n", log);
    free(log);
  }
  if (err != CL_SUCCESS)
    die("glBuildProgram", err);
  start_step = create_kernel("start_step", prog, did);
  end_step = create_kernel("end_step", prog, did);
  keep_on_wire = create_kernel("keep_on_wire", prog, did);
  bead_cols = create_kernel("bead_cols", prog, did);
  cmdq = clCreateCommandQueue(ctx, did, CL_QUEUE_PROFILING_ENABLE, &err);
  if (err != CL_SUCCESS)
    die("clCreateCommandQueue", err);
}


static void init_groups(void) {
  srand48(100);
  groups = xmalloc(n_groups * sizeof(*groups));
  for (int i = 0; i < n_groups; i++) { 
    struct group *g = &groups[i];
    g->wire = (struct wire) {{0.0f, 0.0f}, 0.8f};
    float r = 0.1f;
    float rot = 0.0f;
    for (int j = 0; j < N_BEADS; j++) {
      g->beads_radius[j] = r;
      g->beads_mass[j] = (float) M_PI * r * r; 
      g->beads_pos[j].x = g->wire.pos.x + g->wire.radius * cosf(rot);
      g->beads_pos[j].y = g->wire.pos.y + g->wire.radius * sinf(rot);
      rot += (float) M_PI / N_BEADS;
      r = 0.05f + drand48() * 0.1f;
    }
  }
}

static void update_kernel(cl_kernel kernel) {
  cl_event ev;
  cl_int err = clEnqueueNDRangeKernel(cmdq, kernel, 1, NULL,
      (size_t[]) {n_groups * N_BEADS}, NULL, 0, NULL, &ev);
  if (err != CL_SUCCESS)
    die("clEnqueueTask", err);
  err = clWaitForEvents(1, &ev);
  if (err != CL_SUCCESS)
    die("clWaitForEvents", err);
  cl_ulong start, end;
  err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, 8, &start, NULL);
  if (err != CL_SUCCESS)
    die("clGetEventProfilingInfo", err);
  err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, 8, &end, NULL);
  if (err != CL_SUCCESS)
    die("clGetEventProfilingInfo", err);
  elapsed += end - start;
  clReleaseEvent(ev);
}

#define N_STEPS 100

static void update_sim(void) {
  for (int i = 0; i < N_STEPS; i++) {
    update_kernel(start_step);
    update_kernel(keep_on_wire);
    update_kernel(bead_cols);
    update_kernel(end_step);
  }
  int err = clEnqueueReadBuffer(cmdq, groups_mem, CL_TRUE, 0, 
      n_groups * sizeof(*groups), groups, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    die("clEnqueueReadBuffer", err);
}

static FILE *open_csv(int i) {
  char buf[64]; 
  sprintf(buf, "out%06d.csv", i); 
  FILE *csv = fopen(buf, "wb");
  if (!csv)
    die("fopen", errno);
  return csv;
}

static void open_all(void) {
    csvs = xmalloc(n_groups * sizeof(FILE *)); 
    for (int i = 0; i < n_groups; i++) {
      char buf[64]; 
      sprintf(buf, "out%03d.csv", i); 
      csvs[i] = open_csv(i); 
    }
}

static void print_header(FILE *csv) {
    fprintf(csv, "f,t,x,y,r\n");
}

static void print_sim_one(FILE *csv, struct group *g, int frame) {
  for (int i = 0; i < N_BEADS; i++) {
    fprintf(csv, "%d,%d,%f,%f,%f\n", frame, 0, 
        g->beads_pos[i].x, g->beads_pos[i].y, g->beads_radius[i]);
  }
  fprintf(csv, "%d,%d,%f,%f,%f\n", frame, 1, 
      g->wire.pos.x, g->wire.pos.y, g->wire.radius);
}

static void print_sim_all(int frame) {
  for (int i = 0; i < n_groups; i++) 
    print_sim_one(csvs[i], &groups[i], frame);
}

static void print_time(const char *msg, cl_ulong ul) {
  printf("%s: ", msg);
  cl_ulong n = 1;
  for (; ul / n; n *= 1000);
  if (ul != 0) 
    n /= 1000;
  printf("%lu", ul / n % 1000);
  while (n /= 1000, n) 
    printf(",%03lu", ul / n % 1000);
  printf(" ns\n");
}

int main(int argc, char **argv) {
  parse_args(argc, argv);
  init_groups();
  init_cl();
  if (ends_only) {
    struct group *init = xmalloc(n_groups * sizeof(*groups));
    memcpy(init, groups, n_groups * sizeof(*groups));
    int f;
    for (f = 0; f < 10 * FPS; f++) 
      update_sim();
    for (int i = 0; i < n_groups; i++) {
      FILE *csv = open_csv(i);
      print_header(csv);
      print_sim_one(csv, &init[i], 0);
      print_sim_one(csv, &groups[i], 1);
      fclose(csv);
    }
    free(init);
  } else {
    open_all();
    for (int i = 0; i < n_groups; i++)
      print_header(csvs[i]);
    int f;
    for (f = 0; f < 10 * FPS; f++) {
      print_sim_all(f);
      update_sim();
    }
    print_sim_all(f);
    for (int i = 0; i < n_groups; i++) 
      fclose(csvs[i]);
    free(csvs);
  }
  free(groups);
  print_time("sum", elapsed); 
  print_time("mean", elapsed / n_groups); 
  return 0;
}
