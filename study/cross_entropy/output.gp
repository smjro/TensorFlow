set terminal x11
set xrange [0:300]
set yrange [0:1]
set xlabel "epoch"
set ylabel "cost"

plot "./data.dat" u 1:2 w l
pause -1 "hit"

