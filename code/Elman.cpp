#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace::std;

int ind2d(int i, int j, int n_i, int n_j){
	return i*n_j + j;
}

double s(double x){
	return tanh(x);
}

double ds(double x){
	return 1. - pow(s(x),2);
}

/**double mat_prod(double A[],double B[], int k, int l, int m_a_n_b_start, int m_a_n_b_end){
	
	double output= 0.;

			for(int i=m_a_n_b_start;i<=m_a_n_b_end;i++){
				output += A[k][i]*B[i,l];
			}

	return output;

}**/

double mat_vec_prod(double && A[][], double & B[], int k, int m_a_n_b_start, int m_a_n_b_end){
	double output= 0.;

			for(int i=m_a_n_b_start;i<=m_a_n_b_end;i++){
				output += A[k][i]*B[i];
			}

	return output;
}	


int main(){
	cout << "test" << endl;

	const int N_i = 1;
	const int N_h = 10;
	const int N_o = 1;

	const int N_t = 1000;

	double h[N_h];
	double c[N_h];
	double h_old[N_h];
	double c_old[N_h];
	double I[N_i];
	double o[N_o];



	for(int k=0;k<N_h;k++){
		h[k] = 0.;
		c[k] = 0.;
	}

	for(int k=0;k<N_i;k++){
		I[k] = 0.;
	}

	for(int k=0;k<N_o;k++){
		o[k] = 0.;
	}

	double W_h[N_h][N_h + N_i + 1];
	double W_o[N_o][N_h + 1];

	double I_pregen[N_t][N_i];

	for(int k=0;k<N_t;k++){

		I_pregen[k][0] = sin(k*0.125*2.*M_PI);
	}

	for(int t=0;t<N_t;t++){

		

		for(int k=0;k<N_o;k++){
			o[k] =  s(mat_vec_prod(W_o,h,k,0,N_h-1) + W_o[k][N_h]);
		}

		for(int k = 0;k<N_h;k++){
			h[k] = s(mat_vec_prod(W_h,c,k,0,N_h-1) + mat_vec_prod(W_h,I,k,N_h,N_h+N_i-1) + W_h[k][N_h + N_i]);
			c[k] = h[k]

		}
		I = I_pregen[t];
		

	}

	

	return 0;
}

