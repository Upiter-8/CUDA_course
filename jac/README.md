# Jacobi3d

Для сборки проекта надо выполнить

```bash
make
```

Файлы jaccpu и jacgpu отвечают за запуск на cpu и gpu соответсвенно

Метод также можно запускать, указывая размер матрицы

```bash
./jaccpu 900
./jacgpu 900
```

Сравнение проводится командой, где также можно указать размер матрицы

```bash
./compare 900
```

Вывод jacgpu при размере матрицы 900
```bash
 IT =    1   EPS =  2.6980000E+03
 IT =    2   EPS =  1.3495000E+03
 IT =    3   EPS =  5.2458333E+02
 IT =    4   EPS =  3.3714352E+02
 IT =    5   EPS =  2.7466667E+02
 IT =    6   EPS =  2.3302469E+02
 IT =    7   EPS =  1.9677308E+02
 IT =    8   EPS =  1.6295002E+02
 IT =    9   EPS =  1.3928905E+02
 IT =   10   EPS =  1.2331360E+02
 IT =   11   EPS =  1.1221165E+02
 IT =   12   EPS =  1.0546532E+02
 IT =   13   EPS =  9.8133253E+01
 IT =   14   EPS =  9.1255019E+01
 IT =   15   EPS =  8.4403267E+01
 IT =   16   EPS =  7.8237112E+01
 IT =   17   EPS =  7.2296906E+01
 IT =   18   EPS =  6.7634362E+01
 IT =   19   EPS =  6.3714148E+01
 IT =   20   EPS =  6.0632024E+01
 Jacobi3D Benchmark Completed.
 Size            =  900 x  900 x  900
 Iterations      =                 20
 Time in seconds =               1.88
 Operation type  =     double
 GPU Device: Tesla P100-SXM2-16GB
 Total Global Memory: 16280 MB
 GPU Memory used =     11123.66 MB
 END OF Jacobi3D Benchmark
```

Реализация на CUDA работает быстрее, чем параллельная версия на cpu
