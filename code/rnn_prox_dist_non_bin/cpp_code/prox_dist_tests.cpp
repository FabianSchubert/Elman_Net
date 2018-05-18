#include <iostream>
#include <cmath>
#include "./parameters.hpp"
#include <fstream>
#include <random>
#include <ctime>

using namespace std;
using namespace parameters;

double sigm(double x)
{
	return (tanh(x/2.)+1.)/2.;
}

double M(double y_d)
{
	return f_0_d + a_d*sigm((y_d - theta_d)/g_d);
}

double T(double y_d)
{
	return t_0_d + a_t*sigm((y_d - t_theta_d)/g_t_d);
}

double f_act(double y_p, double y_d)
{
	return M(y_d)*sigm((y_p - T(y_d))/g_p);
}

double h(double x, double y, double y_mean,double x_0)
{
	double phi=(-x+tanh(2.*0.958*x/x_0)*x_0/0.958);
	return phi*(y-y_mean);
}


int main()
{	
	
	time_t timer;

	mt19937 e(time(&timer));

	normal_distribution<> n_dist(0.0,1.0);

	double w_p = 0.01;
	double w_d = 0.01;

	double y_p, y_d, y_p_mean, y_d_mean, y_post, x_p, x_d;
	double M,T;

	y_p_mean = 0.0;
	y_d_mean = 0.0;

	double * w_p_rec = new double[n_t];
	double * w_d_rec = new double[n_t];

	double * y_p_rec = new double[n_t];
	double * y_d_rec = new double[n_t];

	

	for(int t=0 ; t < n_t; t++)
	{
		y_p = n_dist(e);
		y_d = n_dist(e);

		x_p = y_p*w_p;
		x_d = y_d*w_d;

		M = M(y_d);
		T = T(y_d);

		


	}



	/*
	double y_p = 0.5;
	double y_d = 0.5;

	int xy_dim[2] = {100,100};
	double f[xy_dim[0]][xy_dim[1]];

	for(int i = 0; i < xy_dim[0]; i++)
	{
		for(int j = 0; j < xy_dim[1]; j++)
		{
			f[i][j] = f_act(1.*i/xy_dim[0],1.*j/xy_dim[1]);
		}
	}

	ofstream datafile;
	datafile.open("data.csv");

	for(int i = 0; i < xy_dim[0]; i++)
	{
		for(int j = 0; j < xy_dim[1]; j++)
		{
			datafile << f[i][j];
			if(j < (xy_dim[1] - 1))
			{
				datafile << ",";
			}
		}
		
		if(i < (xy_dim[0] - 1))
		{
			datafile << "\n";
		}

		datafile << flush;
	}

	datafile.close();
	*/

	delete [] w_p_rec;
	delete [] w_d_rec;

	return 0;
}


/*#include <iostream>
#include <fstream>
#include <armadillo>

using namespace std;



double f_act(double f_0_d, double a_d, double theta_d, double g_d, double t_0_d, double a_t, double t_theta_d, double g_t_d, double g_p,double y_p, double y_d){
	double M = f_0_d +  a_d
	return M;
}


int main(){

	

	return 0;
}
*/