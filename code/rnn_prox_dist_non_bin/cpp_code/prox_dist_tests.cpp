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
	return m_0 + a_m*sigm((y_d - theta_m)/s_m);
}

double T(double y_d)
{
	return a_t*sigm((y_d - theta_t)/s_t);
}

double f_act(double y_p, double y_d)
{
	return M(y_d)*sigm((y_p + T(y_d) - theta_p)/s_p);
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

	exponential_distribution<> n_dist(1./.5);

	double w_p = 0.01;
	double w_d = 0.01;

	double y_p, y_d, y_p_mean, y_d_mean, y_post, x_p, x_d;
	double m,t;

	y_p_mean = 0.0;
	y_d_mean = 0.0;

	double * w_p_rec = new double[n_t];
	double * w_d_rec = new double[n_t];

	double * y_p_rec = new double[n_t];
	double * y_d_rec = new double[n_t];

	double * y_post_rec = new double[n_t];

	for(int k = 0; k < n_t; k++)
	{
		y_p = n_dist(e);
		y_d = n_dist(e);

		x_p = y_p*w_p;
		x_d = y_d*w_d;

		m = M(y_d);
		t = T(y_d);

		y_post = f_act(x_p, x_d);

		y_p_mean += mu_mean_act*(y_p - y_p_mean);
		y_d_mean += mu_mean_act*(y_d - y_d_mean);
		
		w_d += mu_learn*h(x_d - , x_d, y_d_mean, 1.0);
		w_p += mu_learn*h(t, x_p + t - theta_p, y_p_mean, 1.0);

		w_p_rec[k] = w_p;
		w_d_rec[k] = w_d;

		y_p_rec[k] = y_p;
		y_d_rec[k] = y_d;
		y_post_rec[k] = y_post;


	}

	ofstream datafile_act;
	datafile_act.open("data_act.csv");

	ofstream datafile_weights;
	datafile_weights.open("data_weights.csv");

	for(int k = 0; k < n_t; k++)
	{
		datafile_act << y_p_rec[k] << "," << y_d_rec[k] << "," << y_post_rec[k];

		if(k < n_t-1)
		{
			datafile_act << "\n";
		}
		datafile_act << flush;
	}

	for(int k = 0; k < n_t; k++)
	{
		datafile_weights << w_p_rec[k] << "," << w_d_rec[k];

		if(k < n_t-1)
		{
			datafile_weights << "\n";
		}
		datafile_weights << flush;
	}

	datafile_act.close();
	datafile_weights.close();
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

	delete [] y_p_rec;
	delete [] y_d_rec;

	delete [] y_post_rec;

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