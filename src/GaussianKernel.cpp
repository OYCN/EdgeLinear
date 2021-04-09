/*
URL: https://blog.csdn.net/dcrmg/article/details/52304446#
*/

#include "iostream"
#include "math.h"
 
using namespace std;
 
//******************高斯卷积核生成函数*************************
//第一个参数gaus是一个指向含有3个double类型数组的指针；
//第二个参数size是高斯卷积核的尺寸大小；
//第三个参数sigma是卷积核的标准差
//*************************************************************
void GetGaussianKernel(double **gaus, const int size,const double sigma);
 
int main(int argc,char *argv[])  
{
	int size=5; //定义卷积核大小
	double **gaus=new double *[size];
	for(int i=0;i<size;i++)
	{
		gaus[i]=new double[size];  //动态生成矩阵
	}
	// cout<<"尺寸 = 3*3，Sigma = 1，高斯卷积核参数为："<<endl;
	// GetGaussianKernel(gaus,3,1); //生成3*3 大小高斯卷积核，Sigma=1；	
	cout<<"尺寸 = 5*5，Sigma = 10，高斯卷积核参数为："<<endl;
	GetGaussianKernel(gaus,5,10); //生成5*5 大小高斯卷积核，Sigma=1；	
	// system("pause");
	return 0;
}
 
//******************高斯卷积核生成函数*************************
void GetGaussianKernel(double **gaus, const int size,const double sigma)
{
	const double PI=4.0*atan(1.0); //圆周率π赋值
	int center=size/2;
	double sum=0;
	for(int i=0;i<size;i++)
	{
		for(int j=0;j<size;j++)
		{
			gaus[i][j]=(1/(2*PI*sigma*sigma))*exp(-((i-center)*(i-center)+(j-center)*(j-center))/(2*sigma*sigma));
			sum+=gaus[i][j];
		}
	}
 
	for(int i=0;i<size;i++)
	{
		for(int j=0;j<size;j++)
		{
			gaus[i][j]/=sum;
			cout<<gaus[i][j]<<"  ";
		}
		cout<<endl<<endl;
	}
	return ;
}