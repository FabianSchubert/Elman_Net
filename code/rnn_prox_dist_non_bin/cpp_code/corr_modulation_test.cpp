#include <iostream>
#include <cmath>
#include <fstream>
#include <random>
#include <ctime>
#include "./parameters_corr_modulation.hpp"

using namespace std;
using namespace parameters;

double sigm(double x){
	return (tanh(x/2.) + 1.)/2.;
}

double h(double x, double y, double y_mean, double x_0)
{
	double phi=(-x+tanh(2.*0.958*x/x_0)*x_0/0.958);
	return phi*(y-y_mean);
}

double mu_mod(double x1, double x2){
	return (x1 - x2)*(x1 - x2);
}

int id(int x1, int x2, int n_x2){
	return x1*n_x2 + x2;
}

int main(){

	time_t timer;

	mt19937 e(time(&timer));

	normal_distribution<> n_dist(0.,1.);

	double w_p [n_p];
	double w_d [n_d];

	double y_p [n_p];
	double y_d [n_d];
	double y_p_mean [n_p];
	double y_d_mean [n_d];
	double y_post;
	double x_p [n_p];
	double x_d [n_d];

	for(int i=0; i<n_p; i++){
		w_p[i] = 0.01;
	}
	
	for(int i=0; i<n_d; i++){
		w_d[i] = 0.01;
	}

	double * w_p_rec = new double[n_t*n_p];
	double * w_d_rec = new double[n_t*n_d];

	double * y_p_rec = new double[n_t*n_p];
	double * y_d_rec = new double[n_t*n_d];

	double * y_p_mean_rec = new double[n_t*n_p];
	double * y_d_mean_rec = new double[n_t*n_d];

	double * y_post_rec = new double[n_t];

	double alpha, beta; //used for generating correlated input between distal and first proximal input.

	for(int t = 0; t < n_t; t++){

		for(int k = 0; k < n_d; k++){
			y_d[k] = n_dist(e)*std_d[k];
			y_d_rec[id(t,k,n_d)] = y_d[k];
		}

		alpha = 1./(std_d[0]*(1./corr_p_d - 1.) + 1.);
		beta = std_p[0]/sqrt(std_d[0]*std_d[0]*alpha*alpha + (1.-alpha)*(1.-alpha));

		//y_p[0] = y_d[0];
		y_p[0] = beta*( y_d[0]*alpha + (1.-alpha)*n_dist(e) );
		y_p_rec[id(t,0,n_p)] = y_p[0];

		for(int k = 1; k < n_p; k++){
			y_p[k] = n_dist(e)*std_p[k];
			y_p_rec[id(t,k,n_p)] = y_p[k];
		}
		
		for(int k = 0; k < n_p; k++){
			y_p_mean[k] += mu_mean_act*(y_p[k] - y_p_mean[k]);
			y_p_mean_rec[id(t,k,n_p)] = y_p_mean[k];
		}

		for(int k = 0; k < n_d; k++){
			y_d_mean[k] += mu_mean_act*(y_d[k] - y_d_mean[k]);
			y_d_mean_rec[id(t,k,n_d)] = y_d_mean[k];
		}


	}

	//Write proximal presynaptic activity
	ofstream datafile_act_prox;
	datafile_act_prox.open("data_act_prox.csv");

	datafile_act_prox << "y_p" << endl;
	for(int t = 0; t < n_t; t++)
	{
		
		for(int k = 0; k < (n_p - 1); k++){
			datafile_act_prox << y_p_rec[id(t,k,n_p)] << ",";

		}

		if(t < (n_t - 1)){
			datafile_act_prox << y_p_rec[id(t,n_p-1,n_p)] << "\n";	
		} else{
			datafile_act_prox << y_p_rec[id(t,n_p-1,n_p)];
		}
		
		datafile_act_prox << flush;
	}

	datafile_act_prox.close();

	//Write distal presynaptic activity
	ofstream datafile_act_dist;
	datafile_act_dist.open("data_act_dist.csv");

	datafile_act_dist << "y_d" << endl;
	for(int t = 0; t < n_t; t++)
	{
		
		for(int k = 0; k < (n_d - 1); k++){
			datafile_act_dist << y_d_rec[id(t,k,n_d)] << ",";

		}

		if(t < (n_t - 1)){
			datafile_act_dist << y_d_rec[id(t,n_d-1,n_d)] << "\n";	
		} else{
			datafile_act_dist << y_d_rec[id(t,n_d-1,n_d)];
		}
		
		datafile_act_dist << flush;
	}

	datafile_act_dist.close();




	delete [] w_p_rec;
	delete [] w_d_rec;

	delete [] y_p_rec;
	delete [] y_d_rec;

	delete [] y_p_mean_rec;
	delete [] y_d_mean_rec;

	delete [] y_post_rec;


	return 0;
}




