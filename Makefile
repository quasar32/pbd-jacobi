.PHONY: clean

pbd: pbd.o
	gcc $^ -o $@ -lm -lOpenCL -fsanitize=undefined

profile: profile.c
	gcc $< -o $@ -D _GNU_SOURCE

pbd.o: pbd.c
	gcc $< -o $@ -c -D CL_TARGET_OPENCL_VERSION=120 -fsanitize=undefined

vid: vid.o glad/src/gl.o
	gcc $^ -o $@ -lSDL2main -lSDL2 -lm -lswscale \
		-lavcodec -lavformat -lavutil -lx264 
	
vid.o: vid.c
	gcc $< -o $@ -c -Iglad/include

glad/src/gl.o: glad/src/gl.c
	gcc $< -o $@ -c -Iglad/include 

clean:
	rm glad/src/gl.o *.o pbd vid *.mp4 *.csv -f profile
