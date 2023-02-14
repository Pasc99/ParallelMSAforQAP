#pragma once
#include "SMSA.h"


SMSA::SMSA() :
	num_trial(30), N_state(6), cost_global(INT32_MAX), cost_mean(0.0), cost_worst(0),
	R_error(0.0), cost_1trial(INT32_MAX)
{
	choose_instance();
	read_trialset();
	initialize();
	success = 0;
}

SMSA::SMSA(const int& instance_num, const int& num_trial, const int& N_state)
{
	choose_instance_auto(instance_num);
	this->num_trial = num_trial;
	this->N_state = N_state;
	//print_DF(Distance, Flow);
	initialize();
}

SMSA::~SMSA()
{
}

void SMSA::reset()
{
	cost_best.assign(N_state, INT32_MAX);
}

void SMSA::initialize()
{
	success = 0;
	sol_best.assign(N_state, std::vector<int>(dim, 0));
	delta_cost.assign(N_state, 0);
	prob_accept.assign(N_state, 0);
	sol.clear();
	sol.resize(N_state, std::vector<int>(dim));
	cost.clear();
	cost.resize(N_state);
	cost_best.assign(N_state, INT32_MAX);
	s1.assign(N_state, 0);
	s2.assign(N_state, 0);
}

// Calculate the best, the worst and the mean of the final costs from all trials
void SMSA::result()
{
	cost_mean = (double)std::accumulate(f_best_all.begin(), f_best_all.end(), 0LL);
	cost_mean = (double)cost_mean / f_best_all.size();

	cost_global = *min_element(f_best_all.begin(), f_best_all.end());
	cost_worst = *max_element(f_best_all.begin(), f_best_all.end());

	int trial_size = num_trial;
	for (int i = 0; i < trial_size; i++) {
		R_error += (double)f_best_all[i] / (double)answer - 1.0;
	}
	R_error = R_error * (100.0 / (double)trial_size);

	success = (success * (100 / (double)trial_size));
}

void SMSA::output_log(const std::string& method)
{
	file_dir = "out/result_log(" + method + ")_" + getDateStr() + ".txt";
	record_score.open(file_dir.c_str(), std::ios::app);
}

void SMSA::permutation(std::vector<int>& sol, const int& dim)
{
	std::mt19937_64 rng(std::random_device{}());
	for (int i = 0; i < dim; i++)
		sol[i] = i;
	std::shuffle(sol.begin(), sol.end(), rng);
}

void SMSA::swap_2opt()
{
	int L = N_state;
	for (int i = 0; i < L; i++) {
		s1[i] = rnd_int(0, dim - 1);
		s2[i] = rnd_int(0, dim - 1);
		while (s1[i] == s2[i]) { 
			s1[i] = rnd_int(0, dim - 1);
			s2[i] = rnd_int(0, dim - 1);
		}
	}
}


void SMSA::SA_accept()
{
	sol_next = sol;
	int& L = N_state;
	for (int i = 0; i < L; i++) {
		delta_cost[i] = calculate_delta_cost(sol[i], s1[i], s2[i]);
		prob_accept[i] = exp(-(double)delta_cost[i] / T_curr);

		sol_next[i][s1[i]] = sol[i][s2[i]];
		sol_next[i][s2[i]] = sol[i][s1[i]];

		if (delta_cost[i] < 0) {
			sol[i] = sol_next[i];
			cost[i] += delta_cost[i]; 
		}
		else if (rnd(0, 1) <= prob_accept[i]) {
			sol[i] = sol_next[i];
			cost[i] += delta_cost[i];
		}

		if (cost_best[i] > cost[i]) {
			sol_best[i] = sol[i];
			cost_best[i] = cost[i];
		}
	}
}

void SMSA::SolveQAP()
{
	output_log(m_method);
	record_score << "==============================================\n";
	record_score << "Instance: " << instance << "\nT_0 = " << T_0
		<< ", T_end = " << T_end << ", alpha = " << alpha << "\n";
	record_score << "Number of repetition: " << num_re << "\n";
	record_score << "Number of trials: " << num_trial << "\n";
	record_score << "Number of serial Markov chains: " << N_state << "\n";
	record_score << "Starting time: " << getTimeStr() << std::endl;
	record_score << "Method: " << m_method << std::endl;

	std::cout << "QAP instance: " << instance;
	std::cout << "\nKnown optimal cost: " << answer;
	std::cout << "\nNumber of trials: " << num_trial;
	std::cout << "\nNumber of initial states: " << N_state << "\n";
	std::cout << "Method: " << m_method << std::endl;
	Test_trial();
	
	int& trial_size = num_trial;
	int& L = N_state;

	auto start = std::chrono::high_resolution_clock::now();

	for (int trial = 0; trial < trial_size; trial++) {

		
		for (int i = 0; i < L; i++)
		{
			permutation(sol[i], dim);
			cost[i] = calculate_cost(sol[i]);
		}
			
		int& K_size = num_re; 
		double& alpha_size = alpha;
		double& T_end_size = T_end;

		for (T_curr = T_0; T_curr >= T_end_size; T_curr *= alpha_size)
		{
			cost_previous = cost;
			for (int R = 0; R < K_size; R++) {
				swap_2opt();
				SA_accept();
			}
		} // End of one trial

		cost_1trial = *min_element(cost_best.begin(), cost_best.end());
		display_progress(trial, num_trial);

		success_counter(cost_1trial);
		f_best_all.push_back(cost_1trial);

		reset();
	}


	result();
	display_progress(num_trial, num_trial);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	double duration_per_trial_s = duration * 0.000001 / (double)trial_size;

	record_score << "Success rate: " << success << "%" << std::endl
		<< "fbest: " << cost_global << std::endl
		<< "mean:" << cost_mean << std::endl
		<< "relative error: " << R_error << "%" << std::endl
		<< "worst: " << cost_worst << std::endl
		<< "CPU time per trial: " << duration_per_trial_s << "[s]\n";
	record_score << "Ending time: " << getTimeStr() << "\n";
	record_score << "==============================================\n\n" << std::endl;
	record_score.close();

	std::cout << "\nSuccess rate: " << success << std::endl;
	std::cout << "CPU time per trial: " << duration_per_trial_s << "[s]\n" << std::endl;
}

void SMSA::Test_trial()
{
	std::cout << "\nPerforming test trial...\n";
	int& L = N_state;
	auto start = std::chrono::high_resolution_clock::now();
	
	for (int i = 0; i < L; i++)
	{
		permutation(sol[i], dim);
		cost[i] = calculate_cost(sol[i]);
	}
	
	int& K_size = num_re;
	double& alpha_size = alpha;
	double& T_end_size = T_end;

	for (T_curr = T_0; T_curr >= T_end_size; T_curr *= alpha_size)
	{
		cost_previous = cost;
		for (int R = 0; R < K_size; R++) {
			swap_2opt();
			SA_accept();
		}
	} // End of one trial

	auto iter = std::min_element(cost_best.begin(), cost_best.end());
	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "Test run CPU time: " << duration * 0.000001 << "[s]\n";
	std::cout << "Test run finest cost: " << *iter << std::endl;
	std::cout << "Test run finest solution: ";
	for (int i = 0; i < dim; i++) {
		std::cout << sol[std::distance(cost_best.begin(), iter)][i] << " ";
	}
	
	reset();

	// Estimate ETA
	auto eta = std::chrono::system_clock::now() + std::chrono::microseconds(duration * num_trial);
	auto in_time_t = std::chrono::system_clock::to_time_t(eta);
	struct tm tstruct {};
	localtime_s(&tstruct, &in_time_t);
	std::stringstream tmstr;
	tmstr << std::put_time(&tstruct, "%Y-%m-%d %X");
	std::cout << "\n\nETA: " << tmstr.str() << "\n";
}