#include "PMSA.h"

PMSA::PMSA() :
	num_trial(30), N_state(6), cost_global(INT32_MAX), cost_mean(0.0), cost_worst(0),
	R_error(0.0), cost_1trial(INT32_MAX)
{
	//choose_instance();
}

PMSA::PMSA(const int& instance_num, const int& num_trial, const int& N_state)
{
	choose_instance_auto(instance_num);
	this->num_trial = num_trial;
	this->N_state = N_state;
	//print_DF(Distance, Flow);
	initialize();
}

PMSA::~PMSA()
{

}

void PMSA::reset()
{
}

void PMSA::initialize()
{
	success = 0;
}

// Calculate the best, the worst and the mean of the final costs from all trials
void PMSA::result()
{
	cost_mean = (double)std::accumulate(f_best_all.begin(), f_best_all.end(), 0LL);
	cost_mean = (double)cost_mean / f_best_all.size();

	cost_global = *min_element(f_best_all.begin(), f_best_all.end());
	cost_worst = *max_element(f_best_all.begin(), f_best_all.end());

	int& trial_size = num_trial;
	for (int i = 0; i < num_trial; i++) {
		R_error += (double)f_best_all[i] / (double)answer - 1.0;
	}
	R_error = R_error * (100.0 / (double)trial_size);

	success = (success * (100 / (double)trial_size));
}

double PMSA::rnd(double min_arg, double max_arg)
{
	static thread_local std::mt19937_64 rng(std::random_device{}());
	std::uniform_real_distribution<> uni(min_arg, max_arg);
	return uni(rng);
}

int PMSA::rnd_int(int min_arg, int max_arg)
{
	static thread_local std::mt19937_64 rng(std::random_device{}());
	std::uniform_int_distribution<> uni((int)min_arg, (int)max_arg);
	return uni(rng);
}

void PMSA::output_log(const std::string& method)
{
	file_dir = "out/result_log(" + method + ")_" + getDateStr() + ".txt";
	record_score.open(file_dir.c_str(), std::ios::app);
}

void PMSA::permutation(std::vector<int>& sol, const int& dim)
{
	static thread_local std::mt19937_64 rng(std::random_device{}());
	for (int i = 0; i < dim; i++)
		sol[i] = i;
	std::shuffle(sol.begin(), sol.end(), rng);
}

int PMSA::calculate_cost(std::vector<int>& X)
{
	int cost = 0;
	int N_size = dim;
#pragma omp parallel for reduction(+:cost)
	for (int i = 0; i < N_size; i++) {
		for (int j = 0; j < N_size; j++) {
			if (i != j) {
				cost += Distance[i][j] * Flow[X[i]][X[j]];
			}
		}
	}
	return cost;
}

int PMSA::calculate_delta_cost(std::vector<int>& sol, int& s1, int& s2)
{
	int delta_cost = (Flow[sol[s1]][sol[s2]] - Flow[sol[s2]][sol[s1]])* (Distance[s2][s1] - Distance[s1][s2]);
#pragma omp parallel for reduction(+:delta_cost)
	for (int i = 0; i < dim; i++) {
		if (i != s1 && i != s2) {
			delta_cost += (Flow[sol[i]][sol[s1]] - Flow[sol[i]][sol[s2]]) * (Distance[i][s2] - Distance[i][s1])
				+ (Flow[sol[s1]][sol[i]] - Flow[sol[s2]][sol[i]]) * (Distance[s2][i] - Distance[s1][i]);
		}
	}
	return delta_cost;
}

std::tuple<int, int> PMSA::swap_2opt()
{
	int s1 = rnd_int(0, dim - 1);
	int s2 = 0;
	do
	{
		s2 = rnd_int(0, dim - 1);
	} while (s1 == s2);
	return {s1, s2};
}

void PMSA::SA_accept(std::vector<int>& sol, std::vector<int>& sol_best, 
	int& cost, int& cost_best, int& s1, int& s2, double& T_now)
{
	std::vector<int> sol_next;
	sol_next.resize(dim);
	sol_next = sol;
	int delta_cost = calculate_delta_cost(sol_next, s1, s2);
	double prob_accept = exp(-(double)delta_cost / T_now);

	sol_next[s1] = sol[s2];
	sol_next[s2] = sol[s1];

	if (delta_cost < 0) {
		sol = sol_next;
		cost += delta_cost;
	}
	else if (rnd(0, 1) <= prob_accept) {
		sol = sol_next;
		cost += delta_cost;
	}

	if (cost_best > cost) {
		sol_best = sol;
		cost_best = cost;
	}
	//Modified: sol, sol_best, cost, cost_best
}

std::tuple<std::vector<int>, int> PMSA::SA_2opt(std::vector<int>& sol)
{
	int cost_curr = calculate_cost(sol);
	int cost_tbest = INT_MAX;
	std::vector<int> sol_tbest;
	sol_tbest = sol;

	//SA starts
	for (double T_now = T_0; T_now >= T_end; T_now *= alpha)
	{
		for (int re = 0; re < num_re; re++)
		{
			auto [s1, s2] = swap_2opt();
			SA_accept(sol, sol_tbest, cost_curr, cost_tbest, s1, s2, T_now);
		}
	}
	return {sol_tbest, cost_tbest};
}

void PMSA::SolveQAP()
{
	output_log(m_method);
	record_score << "==============================================\n";
	record_score << "Instance: " << instance << "\nT_0 = " << T_0
		<< ", T_end = " << T_end << ", alpha = " << alpha << "\n";
	record_score << "Number of repetition: " << num_re << "\n";
	record_score << "Number of trials: " << num_trial << "\n";
	record_score << "Number of parallel Markov chains: " << N_state << "\n";
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
	omp_set_num_threads(L);

	// Timer starts
	auto start = std::chrono::high_resolution_clock::now();

	for (int trial = 0; trial < trial_size; trial++) {

		std::vector<std::vector<int>> sol;
		sol.resize(L, std::vector<int>(dim));
		m_sol_gbest = sol[0];	//Initialize gbest solution with one initial state
		std::vector<int> v_cost_best;
		v_cost_best.resize(L);
		int cost_gbest = INT_MAX;

#pragma omp parallel for
		for (int i = 0; i < L; i++)
		{
			permutation(sol[i], dim);
			std::tie(sol[i], v_cost_best[i]) = SA_2opt(sol[i]);
		}
		// End of one trial

		auto iter_best = std::min_element(v_cost_best.begin(), v_cost_best.end());
		cost_gbest = *iter_best;
		m_sol_gbest = sol[std::distance(v_cost_best.begin(), iter_best)];
		// Display progress bar
		display_progress(trial, num_trial);
		success_counter(cost_gbest);
		f_best_all.push_back(cost_gbest);
	} // End of all trials
		
	// Timer stops
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	double duration_per_trial_s = duration * 0.000001 / (double)trial_size;
	
	result();
	display_progress(num_trial, num_trial);
	
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

void PMSA::Test_trial()
{
	std::cout << "\nPerforming test trial...\n";
	int& L = N_state;
	omp_set_num_threads(L);

	// Timer starts
	auto start = std::chrono::high_resolution_clock::now();
	
	std::vector<std::vector<int>> sol;
	sol.resize(L, std::vector<int>(dim));
	std::vector<int> v_cost_best;
	v_cost_best.resize(L);
	int cost_gbest = INT_MAX;

#pragma omp parallel for
	for (int i = 0; i < L; i++)
	{
		permutation(sol[i], dim);
		std::tie(sol[i], v_cost_best[i]) = SA_2opt(sol[i]);
	}
	auto iter_best = std::min_element(v_cost_best.begin(), v_cost_best.end());
	cost_gbest = *iter_best;

	// Timer stops
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();	
	std::cout << "Test run CPU time: " << duration * 0.000001 << "[s]\n";
	std::cout << "Test run finest cost: " << cost_gbest << std::endl;
	std::cout << "Test run finest solution: ";
	for (int i = 0; i < dim; i++) {
		std::cout << sol[std::distance(v_cost_best.begin(), iter_best)][i] << " ";
	}

	// Estimate ETA
	auto eta = std::chrono::system_clock::now() + std::chrono::microseconds(duration * num_trial);
	auto in_time_t = std::chrono::system_clock::to_time_t(eta);
	struct tm tstruct {};
	localtime_s(&tstruct, &in_time_t);
	std::stringstream tmstr;
	tmstr << std::put_time(&tstruct, "%Y-%m-%d %X");
	std::cout << "\n\nETA: " << tmstr.str() << "\n";
}