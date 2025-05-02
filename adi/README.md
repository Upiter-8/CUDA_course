# Adi3d

Для сборки проекта надо выполнить

```bash
make
```

Файлы adicpu и adigpu отвечают за запуск на cpu и gpu соответсвенно

Метод также можно запускать, указывая размер матрицы

```bash
./adicpu 900
./adigpu 900
```

Сравнение проводится командой, где также можно указать размер матрицы

```bash
./compare 900
```

Вывод adigpu при размере матрицы 900
```bash
 IT =    1   EPS =  1.4977753E+01
 IT =    2   EPS =  7.4833148E+00
 IT =    3   EPS =  3.7388765E+00
 IT =    4   EPS =  2.8020717E+00
 IT =    5   EPS =  2.0999896E+00
 IT =    6   EPS =  1.6321086E+00
 IT =    7   EPS =  1.3979074E+00
 IT =    8   EPS =  1.2004305E+00
 IT =    9   EPS =  1.0395964E+00
 IT =   10   EPS =  9.0896725E-01
 ADI Benchmark Completed.
 Size            =  900 x  900 x  900
 Iterations      =                 10
 Time in seconds =               3.59
 Operation type  =     double
 GPU Device: Tesla P100-SXM2-16GB
 Total Global Memory: 16280 MB
 GPU Memory used =     11123.66 MB
 END OF ADI Benchmark
```

Реализация на CUDA работает быстрее, чем параллельная версия на cpu
