#include <stdio.h>
#include <math.h>

#define startingWeight 0.6
#define startingBias 0.9
#define eta 0.15

// プロトタイプ宣言
double outputValue(double, double);
double sigmoid(double);
double derivative(double);

int main(void)
{
  int i=0;
  double weight, bias, a, delta;
  FILE *fp;

  fp = fopen("data.dat", "w");

  // init
  weight = startingWeight;
  bias = startingBias;

  for(; i<300; i++){
    a = outputValue(weight, bias);
    delta = derivative(a);
    weight += -eta*delta;
    bias += -eta*delta;
    printf("%f\n",a);
    fprintf(fp, "%d %f\n", i, a);
  }
  fclose(fp);

  return 0;
}

double outputValue(double weight, double bias)
{
  return sigmoid(weight+bias);
}

// シグモイド関数
double sigmoid(double z)
{
  return 1/(1+exp(-z));
}

//*******************************
// 二次導関数の微分
//*******************************
// f(a)=a*a/2
// a: シグモイド関数から得られた値
//*******************************
double derivative(double a)
{
  return a*a*(1-a);
}
