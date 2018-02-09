all : main2
main2 : custom_rand.c func_def.c gss_filter.c gss_som.c main4.c pcg_basic.c stat.c custom_rand.h func_def.h gss_filter.h gss_som.h pcg_basic.h stat.h pre_def.h 
	gcc -o main2 custom_rand.c func_def.c gss_filter.c gss_som.c main4.c pcg_basic.c stat.c -lm

