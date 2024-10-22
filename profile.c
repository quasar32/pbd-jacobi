#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/wait.h>
#include <unistd.h>

int main(void) {
  pid_t pid;
  int i;
  int err;
  int stat;
  char *argv[5];
  char buf[16];

  argv[0] = (char *) "./pbd";
  argv[1] = (char *) "-g";
  argv[2] = buf;
  argv[3] = (char *) "-e";
  argv[4] = NULL; 
  for (i = 1; i < 1 << 20; i <<= 1) {
    sprintf(buf,"%d", i);
    err = posix_spawn(&pid, "./pbd", NULL, NULL, argv, environ);
    if (err) {
      errno = err;
      perror("posix_spawn");
      exit(EXIT_FAILURE);
    }
    if (waitpid(pid, &stat, 0) < 0) {
      perror("waitpid");
      exit(EXIT_FAILURE);
    }
  }
  return EXIT_SUCCESS;
}
