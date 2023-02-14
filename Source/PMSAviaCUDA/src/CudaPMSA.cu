#pragma once
#include "CudaPMSA.h"

CudaPMSA::CudaPMSA() :
	num_trial(30), N_state(6), block_size(8), cost_global(INT32_MAX), cost_mean(0.0), cost_worst(0),
	R_error(0.0), cost_1trial(INT32_MAX)
{
	//choose_instance();
}

CudaPMSA::CudaPMSA(const int& instance_num, const int& num_trial, const int& N_state) :
	num_trial(30), N_state(6), block_size(8), cost_global(INT32_MAX), cost_mean(0.0), cost_worst(0),
	R_error(0.0), cost_1trial(INT32_MAX)
{
	choose_instance_auto(instance_num);
	this->num_trial = num_trial;
	this->N_state = N_state;
	//print_DF(Distance, Flow);
	initialize();

}

CudaPMSA::~CudaPMSA()
{
	delete[] FDistance;
	delete[] FFlow;
	//checkCudaErrors(cudaFree());
}

void CudaPMSA::reset()
{
}

void CudaPMSA::initialize()
{
	success = 0;
	FDistance = new int[dim * dim];
	FFlow = new int[dim * dim];
	FlattenArray(FDistance, Distance, dim, dim);
	FlattenArray(FFlow, Flow, dim, dim);
}

// Calculate the best, the worst and the mean of the final costs from all trials
void CudaPMSA::result()
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

void CudaPMSA::output_log(const std::string& method)
{
	file_dir = "out/result_log(" + method + ")_" + getDateStr() + ".txt";
	record_score.open(file_dir.c_str(), std::ios::app);
}

void CudaPMSA::FlattenArray(int* dst, const std::vector<std::vector<int>>& src, int row, int col)
{
	for (size_t i = 0; i < row; i++)
		for (size_t j = 0; j < col; j++)
			dst[i * col + j] = src[i][j];
}

void CudaPMSA::RestoreVector(std::vector<std::vector<int>>& dst, int* src, int row, int col)
{
	for (size_t i = 0; i < row; i++)
		for (size_t j = 0; j < col; j++)
			dst[i][j] = src[i * col + j];
}

void CudaPMSA::SolveQAP()
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

	int& trial_size = num_trial;
	int& total_number_of_threads = N_state;
	size_t grid_size = (N_state - 1) / block_size + 1; // number of blocks per grid

	Test_trial();

	//Timer starts
	auto start = std::chrono::high_resolution_clock::now();

	// Initializing XORWOW
	curandState_t* dev_random;
	checkCudaErrors(cudaMalloc(&dev_random, total_number_of_threads * sizeof(curandState_t)));
	xorwow_init_kernel << <grid_size, block_size >> > (dev_random, time(nullptr), total_number_of_threads);

	int* d_Distance, * d_Flow;
	checkCudaErrors(cudaMalloc((int**)&d_Distance, dim * dim * sizeof(int)));
	checkCudaErrors(cudaMalloc((int**)&d_Flow, dim * dim * sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_Distance, FDistance, dim * dim * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Flow, FFlow, dim * dim * sizeof(int), cudaMemcpyHostToDevice));

	for (int trial = 0; trial < trial_size; trial++) {
		// Stores the best cost/sol of threads on the HOST
		int cost_gbest = INT_MAX;
		int* h_BestSolution = new int[dim];
		int* h_BestCost = new int[N_state];
		std::vector<int> v_cost_best;
		v_cost_best.resize(N_state);

		// Copy data from HOST to DEVICE
		// Not copying initial solutions from trialset but generate them in the kernel
		int* d_BestSolution, * d_BestSolutionArray, * d_BestCost;

		checkCudaErrors(cudaMalloc((int**)&d_BestSolution, dim * sizeof(int)));
		checkCudaErrors(cudaMalloc((int**)&d_BestSolutionArray, dim * total_number_of_threads * sizeof(int)));
		checkCudaErrors(cudaMalloc((int**)&d_BestCost, total_number_of_threads * sizeof(int)));

		// <<<numBlocks, threadsPerBlock>>>
		SolveQAP_final << <grid_size, block_size >> > (dev_random, dim, total_number_of_threads, d_Distance, d_Flow, d_BestSolutionArray, d_BestCost, T_0, T_end, alpha, num_re);
		cudaDeviceSynchronize();
		// End of one trial

		// Copy cost from DEVICE back to HOST in order to find minima
		checkCudaErrors(cudaMemcpy(h_BestCost, d_BestCost, total_number_of_threads * sizeof(int), cudaMemcpyDeviceToHost));

		v_cost_best.assign(h_BestCost, h_BestCost + total_number_of_threads);
		auto iter_best = std::min_element(v_cost_best.begin(), v_cost_best.end());
		cost_gbest = *iter_best;

		// Copy best solution from DEVICE back to HOST
		/*int index = std::distance(v_cost_best.begin(), iter_best);
		AcquireBestSolution << <grid_size, block_size >> > (d_BestSolution, d_BestSolutionArray, dim, index);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemcpy(h_BestSolution, d_BestSolution, dim * sizeof(int), cudaMemcpyDeviceToHost));*/

		success_counter(cost_gbest);
		f_best_all.push_back(cost_gbest);
		//Display progress bar
		display_progress(trial, num_trial);

		// Clean up
		checkCudaErrors(cudaFree(d_BestSolution));
		checkCudaErrors(cudaFree(d_BestSolutionArray));
		checkCudaErrors(cudaFree(d_BestCost));
		delete[] h_BestSolution;
		delete[] h_BestCost;

	} // End of all trials

	//Timer stops
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	//double duration_per_trial_ms = duration * 0.001 / (double)trial_size;
	double duration_per_trial_s = duration * 0.000001 / (double)trial_size;

	result();
	display_progress(num_trial, num_trial);

	// Clean up
	checkCudaErrors(cudaFree(dev_random));
	checkCudaErrors(cudaFree(d_Distance));
	checkCudaErrors(cudaFree(d_Flow));

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

void CudaPMSA::Test_trial()
{
	std::cout << "QAP instance: " << instance;
	std::cout << "\nKnown optimal cost: " << answer;
	std::cout << "\nNumber of trials: " << num_trial;
	std::cout << "\nNumber of initial states: " << N_state << "\n";
	std::cout << "Method: " << m_method << std::endl;

	std::cout << "\nPerforming test trial...\n";

	//Timer starts
	auto start = std::chrono::high_resolution_clock::now();

	int& total_number_of_threads = N_state;
	if (block_size > total_number_of_threads) block_size = total_number_of_threads;
	size_t grid_size = (N_state - 1) / block_size + 1; // number of blocks per grid

	// Stores the best cost/sol of threads on the HOST
	int cost_gbest = INT_MAX;
	int* h_BestSolution = new int[dim];
	int* h_BestCost = new int[N_state];
	std::vector<int> v_cost_best;
	v_cost_best.resize(N_state);

	// Initializing XORWOW
	curandState_t* dev_random;
	checkCudaErrors(cudaMalloc(&dev_random, total_number_of_threads * sizeof(curandState_t)));
	xorwow_init_kernel << <grid_size, block_size >> > (dev_random, time(nullptr), total_number_of_threads);

	// Copy data from HOST to DEVICE
	// Not copying initial solutions from trialset but generate them in the kernel
	int* d_Distance, * d_Flow, * d_BestSolution, * d_BestSolutionArray, * d_BestCost;

	checkCudaErrors(cudaMalloc((int**)&d_Distance, dim * dim * sizeof(int)));
	checkCudaErrors(cudaMalloc((int**)&d_Flow, dim * dim * sizeof(int)));
	checkCudaErrors(cudaMalloc((int**)&d_BestSolution, dim * sizeof(int)));
	checkCudaErrors(cudaMalloc((int**)&d_BestSolutionArray, dim * total_number_of_threads * sizeof(int)));
	checkCudaErrors(cudaMalloc((int**)&d_BestCost, total_number_of_threads * sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_Distance, FDistance, dim * dim * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Flow, FFlow, dim * dim * sizeof(int), cudaMemcpyHostToDevice));

	auto end_init = std::chrono::high_resolution_clock::now();
	auto duration_init = std::chrono::duration_cast<std::chrono::microseconds>(end_init - start).count();
	std::cout << "Test run initialization time: " << duration_init * 0.000001 << "[s]\n";

	// <<<numBlocks, threadsPerBlock>>>
	SolveQAP_final << <grid_size, block_size >> > (dev_random, dim, total_number_of_threads, d_Distance, d_Flow, d_BestSolutionArray, d_BestCost, T_0, T_end, alpha, num_re);
	cudaDeviceSynchronize();
	// End of one trial

	auto end_gpu = std::chrono::high_resolution_clock::now();
	auto duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - end_init).count();
	std::cout << "Test run SA kernel time: " << duration_gpu * 0.000001 << "[s]\n";

	// Copy cost from DEVICE back to HOST in order to find minima
	checkCudaErrors(cudaMemcpy(h_BestCost, d_BestCost, total_number_of_threads * sizeof(int), cudaMemcpyDeviceToHost));

	v_cost_best.assign(h_BestCost, h_BestCost + total_number_of_threads);
	auto iter_best = std::min_element(v_cost_best.begin(), v_cost_best.end());
	cost_gbest = *iter_best;
	int index = std::distance(v_cost_best.begin(), iter_best);
	AcquireBestSolution << <grid_size, block_size >> > (d_BestSolution, d_BestSolutionArray, dim, index);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(h_BestSolution, d_BestSolution, dim * sizeof(int), cudaMemcpyDeviceToHost));

	//Timer stops
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	auto duration_finishup = std::chrono::duration_cast<std::chrono::microseconds>(end - end_gpu).count();
	std::cout << "Test run result collection time: " << duration_finishup * 0.000001 << "[s]\n";
	std::cout << "Test run total CPU time: " << duration * 0.000001 << "[s]\n";
	std::cout << "Test run finest cost: " << cost_gbest << std::endl;
	std::cout << "Test run finest solution: ";
	for (int i = 0; i < dim; i++) {
		std::cout << h_BestSolution[i] << " ";
	}

	// Clean up
	checkCudaErrors(cudaFree(dev_random));
	checkCudaErrors(cudaFree(d_Distance));
	checkCudaErrors(cudaFree(d_Flow));
	checkCudaErrors(cudaFree(d_BestSolution));
	checkCudaErrors(cudaFree(d_BestSolutionArray));
	checkCudaErrors(cudaFree(d_BestCost));
	delete[] h_BestSolution;
	delete[] h_BestCost;

	// Estimate ETA
	auto eta = std::chrono::system_clock::now() + std::chrono::microseconds(duration * num_trial);
	auto in_time_t = std::chrono::system_clock::to_time_t(eta);
	struct tm tstruct {};
	localtime_s(&tstruct, &in_time_t);
	std::stringstream tmstr;
	tmstr << std::put_time(&tstruct, "%Y-%m-%d %X");
	std::cout << "\n\nETA: " << tmstr.str() << "\n";

	// Output specific execution time
	std::string file_dir_test = "out/result_testrun_time(CUDA)_" + getDateStr() + ".txt";
	std::ofstream testrun_time;
	testrun_time.open(file_dir_test.c_str(), std::ios::app);
	testrun_time << "==============================================\n";
	testrun_time << "Instance: " << instance << "\nT_0 = " << T_0
		<< ", T_end = " << T_end << ", alpha = " << alpha << "\n";
	testrun_time << "Number of repetition: " << num_re << "\n";
	testrun_time << "Number of parallel Markov chains: " << N_state << "\n";
	testrun_time << "Method: " << m_method << std::endl;
	testrun_time << std::fixed << std::setprecision(3) << "Test run initialization time: " << duration_init * 0.000001 << "[s]\t\t(" << (double)duration_init / (double)duration * 100 << " %)\n"
		<< "Test run SA kernel time: " << duration_gpu * 0.000001 << "[s]\t\t\t(" << (double)duration_gpu / (double)duration * 100 << " %)\n"
		<< "Test run result collection time: " << duration_finishup * 0.000001 << "[s]\t(" << (double)duration_finishup / (double)duration * 100 << " %)\n"
		<< "Test run total execution time: " << duration * 0.000001 << "[s]\n";
	testrun_time << "Test run finished at: " << getTimeStr() << std::endl;
	testrun_time << "==============================================\n\n" << std::endl;
	testrun_time.close();
}
