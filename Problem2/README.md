make sure the CMakeLists.txt file and all dependancies build

Experiments were conducted on a windows machine.

Commands to run:

cmake -B build -S .

cmake --build build --config Release

build\Release\conv.exe

















































Output For Experiment Run:

starting test
Testing filters on image: image21_in.pgm
GPU Device 0: "Ada" with compute capability 8.9

sdkFindFilePath <image21.pgm> in ./
sdkFindFilePath <image21.pgm> in ./data/
Loaded 'image21_in.pgm', 512 x 512 pixels

Testing regular CUDA convolution:

Testing shared memory CUDA convolution:

Testing sequential CPU convolution:
Wrote './data/image21_emboss_out.pgm'

Testing regular CUDA convolution:

Testing shared memory CUDA convolution:

Testing sequential CPU convolution:
Wrote './data/image21_sharpen_out.pgm'

Testing regular CUDA convolution:

Testing shared memory CUDA convolution:

Testing sequential CPU convolution:
Wrote './data/image21_average_out.pgm'
Testing filters on image: lena_bw_in.pgm
GPU Device 0: "Ada" with compute capability 8.9

sdkFindFilePath <lena_bw.pgm> in ./
sdkFindFilePath <lena_bw.pgm> in ./data/
Loaded 'lena_bw_in.pgm', 512 x 512 pixels

Testing regular CUDA convolution:

Testing shared memory CUDA convolution:

Testing sequential CPU convolution:
Wrote './data/lena_bw_emboss_out.pgm'

Testing regular CUDA convolution:

Testing shared memory CUDA convolution:

Testing sequential CPU convolution:
Wrote './data/lena_bw_sharpen_out.pgm'

Testing regular CUDA convolution:

Testing shared memory CUDA convolution:

Testing sequential CPU convolution:
Wrote './data/lena_bw_average_out.pgm'
Testing filters on image: man_in.pgm
GPU Device 0: "Ada" with compute capability 8.9

sdkFindFilePath <man.pgm> in ./
sdkFindFilePath <man.pgm> in ./data/
Loaded 'man_in.pgm', 512 x 512 pixels

Testing regular CUDA convolution:

Testing shared memory CUDA convolution:

Testing sequential CPU convolution:
Wrote './data/man_emboss_out.pgm'

Testing regular CUDA convolution:

Testing shared memory CUDA convolution:

Testing sequential CPU convolution:
Wrote './data/man_sharpen_out.pgm'

Testing regular CUDA convolution:

Testing shared memory CUDA convolution:

Testing sequential CPU convolution:
Wrote './data/man_average_out.pgm'
Testing filters on image: mandrill_in.pgm
GPU Device 0: "Ada" with compute capability 8.9

sdkFindFilePath <mandrill.pgm> in ./
sdkFindFilePath <mandrill.pgm> in ./data/
Loaded 'mandrill_in.pgm', 512 x 512 pixels

Testing regular CUDA convolution:

Testing shared memory CUDA convolution:

Testing sequential CPU convolution:
Wrote './data/mandrill_emboss_out.pgm'

Testing regular CUDA convolution:

Testing shared memory CUDA convolution:

Testing sequential CPU convolution:
Wrote './data/mandrill_sharpen_out.pgm'

Testing regular CUDA convolution:

Testing shared memory CUDA convolution:

Testing sequential CPU convolution:
Wrote './data/mandrill_average_out.pgm'
Testing filters on image: teapot512_in.pgm
GPU Device 0: "Ada" with compute capability 8.9

sdkFindFilePath <teapot512.pgm> in ./
sdkFindFilePath <teapot512.pgm> in ./data/
Loaded 'teapot512_in.pgm', 512 x 512 pixels

Testing regular CUDA convolution:

Testing shared memory CUDA convolution:

Testing sequential CPU convolution:
Wrote './data/teapot512_emboss_out.pgm'

Testing regular CUDA convolution:

Testing shared memory CUDA convolution:

Testing sequential CPU convolution:
Wrote './data/teapot512_sharpen_out.pgm'

Testing regular CUDA convolution:

Testing shared memory CUDA convolution:

Testing sequential CPU convolution:
Wrote './data/teapot512_average_out.pgm'
Image               Filter         Regular     Shared      Sequential
-----------------------------------------------------------------------
image21.pgm         emboss         8968.32     10377.8     6.78539
image21.pgm         sharpen        14539.3     21790.9     18.5284
image21.pgm         average        7445.16     10078.6     6.73105
lena_bw.pgm         emboss         9130.76     10296.3     6.8685
lena_bw.pgm         sharpen        8080.89     19533.8     18.1826
lena_bw.pgm         average        6319.77     9584.79     6.67334
man.pgm             emboss         9840.24     10141       6.82955
man.pgm             sharpen        15312.1     19814.4     18.5526
man.pgm             average        7281.78     10024.6     6.75016
mandrill.pgm        emboss         9978.84     10345.1     6.82365
mandrill.pgm        sharpen        12869.1     11851       18.1531
mandrill.pgm        average        9858.74     10639       6.89363
teapot512.pgm       emboss         8564        10519.4     6.71092
teapot512.pgm       sharpen        9484.23     19724.9     18.1723
teapot512.pgm       average        9204.49     9436.43     6.67169
test completed, returned