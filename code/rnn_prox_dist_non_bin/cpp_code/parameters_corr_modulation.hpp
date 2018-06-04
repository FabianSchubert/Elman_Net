namespace parameters
{	
	// activation function
    const double x_fix_p = 0.5;
	const double x_fix_d = 0.5;

	// input stats
	const int n_d = 1;
	const int n_p = 10;
	
	double std_d [n_d] = {1.};
	double std_p [n_p] = {1.,1.,1.,1.,1.,1.,1.,1.,1.,1.};
		
	const double corr_p_d = 0.7; // correlation between distal input and first proximal input;
	
	// simulation

	const int n_t = 50000;
	const double mu_learn = 0.001;
	const double mu_mean_act = 0.01;
}