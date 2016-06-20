#include <stdio.h>
#include <math.h>

#define startingWeight 2.0
#define startingBias 2.0
#define eta 0.15

int flag;

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
  weight = startingWeight;  // 重み
  bias = startingBias;      // バイアス

  printf("which type (sigmoid:0, cross entropy:1)\n");
  scanf("%d", &flag);

  for(; i<300; i++){
    a = outputValue(weight, bias); // 出力
    delta = derivative(a);         // 偏微分
    weight += -eta*delta;          // 重みの更新
    bias += -eta*delta;            // バイアスの更新
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
// 微分
//******************************************
// 二次導関数の微分(flag==0)
// f(a)=a*a/2
// a: シグモイド関数から得られた値
// -----------------------------------------
// クロスエントロピーコスト関数の微分(flag==0)
// f(a)=ln(1-a)
//******************************************
double derivative(double a)
{
  if(flag==0) return a*a*(1-a);
  if(flag==1) return a;
}
