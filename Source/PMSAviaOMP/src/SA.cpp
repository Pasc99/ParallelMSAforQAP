#include "SA.h"

SA::SA() :
	num_trial(30), N_state(6), T_curr(0), T_0(0), T_end(0), alpha(0.95),
	num_re(1), dim(100), answer(0), success(0.0) // change default value here
{
	mt.seed(rng());
	std::filesystem::create_directories("out/");
}

SA::~SA()
{

}

double SA::rnd(double min_arg, double max_arg)
{
	std::uniform_real_distribution<> uni(min_arg, max_arg);
	return uni(mt);
}//Mersenne twister

int SA::rnd_int(int min_arg, int max_arg)
{
	std::uniform_int_distribution<> uni((int)min_arg, (int)max_arg);
	return uni(mt);
}//Mersenne twister

// Generate initial state as txt for recording annealing curve
// Not used in the final paper
void SA::generate_initX()
{
	int num_initX;
	std::cout << "Generate initial states for each instance.\n";
	std::cout << "Enter number of trials: ";
	std::cin >> num_trial;
	std::cout << "Enter number fo initial states: ";
	std::cin >> N_state;

	while (1)
	{
		choose_instance_lite();
		//Generate all initial state(trial*state_each_trial)
		//(N_state) for the first recorded trial
		num_initX = num_trial * N_state + N_state;

		//Output to res/XXX_ts.txt
		std::string ts_dir = "res/" + instance + "_ts.txt";
		std::ofstream ofs(ts_dir.c_str());

		//Generate initial solution X {0, 1, ..., N-1}
		std::vector<int> initX;
		initX.reserve(dim);
		for (int i = 0; i < dim; i++)
			initX.emplace_back(i);

		//Shuffle X and output to file for each trial and initial state
		for (int i = 0; i < num_initX; i++) {
			std::shuffle(initX.begin(), initX.end(), mt);
			for (int j = 0; j < dim; j++) {
				ofs << initX[j] << " ";
			}
			ofs << "\n";
		}
		ofs.close();
		if (std::filesystem::exists(ts_dir.c_str()))
			std::cout << "File was generated as \"" << ts_dir.c_str() << "\".\n";
		else
			std::cout << "No file was generated.\n";
	}
}

void SA::generate_initX_auto(const int& instance_num)
{
	int num_initX;

	choose_instance_auto(instance_num);
	//Generate all initial state(trial*state_each_trial)
	//(N_state) for the first recorded trial
	num_initX = num_trial * N_state;

	//Output to res/XXX_ts.txt
	std::string ts_dir = "res/" + instance + "_ts.txt";
	std::ofstream ofs(ts_dir.c_str());

	//Generate initial solution X {0, 1, ..., N-1}
	std::vector<int> initX;
	initX.reserve(dim);
	for (int i = 0; i < dim; i++)
		initX.emplace_back(i);

	//Shuffle X and output to file for each trial and initial state
	for (int i = 0; i < num_initX; i++) {
		std::shuffle(initX.begin(), initX.end(), mt);
		for (int j = 0; j < dim; j++) {
			ofs << initX[j] << " ";
		}
		ofs << "\n";
	}
	ofs.close();
	if (std::filesystem::exists(ts_dir.c_str()))
		std::cout << num_initX << " initial solutions were generated at \""
		<< ts_dir.c_str() << "\".\n\n";
	else
	{
		std::cout << "No file was generated.\n";
		exit(-2);
	}
}

void SA::read_DF()
{
	std::ifstream ifs(std::string("res/" + instance + ".dat"), std::ios::in);

	if (ifs.fail())
	{
		std::cout << "Failed to read DF data!" << std::endl;
		exit(1);
	}

	ifs >> dim;
	Distance.resize(dim, std::vector<int>(dim));
	Flow.resize(dim, std::vector<int>(dim));
	for (size_t i = 0; i < dim; i++)
		for (size_t j = 0; j < dim; j++)
			ifs >> Distance[i][j];
	for (size_t i = 0; i < dim; i++)
		for (size_t j = 0; j < dim; j++)
			ifs >> Flow[i][j];
	ifs.close();
}

// Print DF matrix read from file
void SA::print_DF(const std::vector<std::vector<int>>& D, const std::vector<std::vector<int>>& F)
{
	std::cout << "\nD matrix:\n";
	for (size_t i = 0; i < D.size(); i++)
	{
		for (size_t j = 0; j < D[0].size(); j++)
			std::cout << D[i][j] << " ";
		std::cout << std::endl;
	}
	std::cout << "\nF matrix:\n";
	for (size_t i = 0; i < F.size(); i++)
	{
		for (size_t j = 0; j < F[0].size(); j++)
			std::cout << F[i][j] << " ";
		std::cout << std::endl;
	}
}

void SA::read_trialset()
{
	std::ifstream ifs(std::string("res/" + instance + "_ts.txt"));
	std::vector<char> buff;

	int num_initX = num_trial * N_state;

	buffer_trialset.resize(num_initX, std::vector<int>(dim));

	if (ifs.fail()) {
		std::cout << "Failed to read trialset data!" << std::endl;
		exit(1);
	}
	for (int i = 0; i < num_initX; i++) {
		for (int j = 0; j < dim; j++) {
			ifs >> buffer_trialset[i][j];
		}
	}

	ifs.close();
}

int SA::calculate_cost(std::vector<int>& X)
{
	int cost = 0;
	int N_size = dim;
	for (int i = 0; i < N_size; i++) {
		for (int j = 0; j < N_size; j++) {
			if (i != j) {
				cost += Distance[i][j] * Flow[X[i]][X[j]];
			}
		}
	}
	return cost;
}

int SA::calculate_delta_cost(std::vector<int>& sol, int& s1, int& s2)
{
	int delta_cost = (Flow[sol[s1]][sol[s2]] - Flow[sol[s2]][sol[s1]]) * (Distance[s2][s1] - Distance[s1][s2]);
	for (int i = 0; i < dim; i++) {
		if (i != s1 && i != s2) {
			delta_cost += (Flow[sol[i]][sol[s1]] - Flow[sol[i]][sol[s2]]) * (Distance[i][s2] - Distance[i][s1])
				+ (Flow[sol[s1]][sol[i]] - Flow[sol[s2]][sol[i]]) * (Distance[s2][i] - Distance[s1][i]);
		}
	}
	return delta_cost;
}

void SA::success_counter(int& f_1trial)
{
	if (f_1trial <= answer)
		success++;
}

void SA::set_param(const char* name, double t0_arg, double T_end_arg, int K_arg, uint32_t answer_arg)
{
	instance = name;
	T_0 = t0_arg;
	T_end = T_end_arg;
	num_re = K_arg;
	answer = answer_arg;
	read_DF();
	//print_DF(D, F);
}

void SA::choose_instance()
{
	std::string str;
	std::cout << "Solving QAP via MSA.\n";
	std::cout << "[0]rou15	[1]had20	[2]lipa20a	[3]nug30	[4]tai25a\n";
	std::cout << "[5]tai30a	[6]tai30b	[7]tai40b	[8]tai80a	[9]tai80b\n";
	std::cout << "Enter the index of QAP instance or \"q\" to exit.\n";
	while (1) {
		std::cout << "Choice: ";
		std::cin >> str;
		if (str == "0") {
			set_param("rou15", 50000, 50, 3000, 354210);
			break;
		}
		else if (str == "1") {
			set_param("had20", 300, 0.1, 3000, 6922);
			break;
		}
		else if (str == "2") {
			set_param("lipa20a", 300, 0.1, 3000, 3683);
			break;
		}
		else if (str == "3") {
			set_param("nug30", 50000, 1, 9000, 6124);
			break;
		}
		else if (str == "4") {
			set_param("tai25a", 200000, 500, 6250, 1167256);
			break;
		}
		else if (str == "5") {
			set_param("tai30a", 120000, 2500, 9000, 1706855);
			break;
		}
		else if (str == "6") {
			set_param("tai30b", 270000000, 1500, 9000, 637117113);
			break;
		}
		else if (str == "7") {
			set_param("tai40b", 270000000, 1500, 15000, 637250948);
			break;
		}
		else if (str == "8") {
			set_param("tai80a", 1000000, 100, 60000, 13557864);
			break;
		}
		else if (str == "9") {
			set_param("tai80b", 450000000, 2000, 60000, 818415043);
			break;
		}
		else if (str == "q") {
			exit(1);
		}
		else {
			std::cout << "Invalid value! Please try again\n";
		}
	}
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	std::string s_num_trial, s_stateN;
	std::cout << "Enter the number of trials (Default: " << num_trial << "): ";
	std::getline(std::cin, s_num_trial);
	if (s_num_trial != "\0")
		num_trial = stod(s_num_trial);
	std::cout << "Enter the number of initial states (Default: " << N_state << "): ";
	std::getline(std::cin, s_stateN);
	if (s_num_trial != "\0")
		N_state = stod(s_stateN);
	std::cout << "\nQAP instance: " << instance;
	std::cout << "\nNumber of trials: " << num_trial;
	std::cout << "\nNumber of initial states: " << N_state << std::endl;
}

void SA::choose_instance_auto(const int& instance_num)
{
	std::cout << "--------------------------------------------------\n";
	std::cout << "Solving QAP via Multi-start Simulated Annealing.\n\n";
	if (instance_num == 0)
		set_param("nug16a", 300, 0.1, 3000, 1610);
	else if (instance_num == 1)
		set_param("nug30", 50000, 1, 9000, 6124);
	else if (instance_num == 2)
		set_param("tai25a", 200000, 500, 6250, 1167256);
	else if (instance_num == 3)
		set_param("tai50b", 100000000, 1000, 25000, 458821517);
	else {
		std::cout << "Invalid instance value!\n";
	}
}

void SA::choose_instance_lite()
{
	std::string str;
	std::cout << "[0]rou15	[1]had20	[2]lipa20a	[3]nug30	[4]tai25a\n";
	std::cout << "[5]tai30a	[6]tai30b	[7]tai40b	[8]tai80a	[9]tai80b\n";
	std::cout << "Enter the index of QAP instance or \"q\" to exit.\n";
	while (1) {
		std::cout << "Choice: ";
		std::cin >> str;
		if (str == "0") {
			instance = "rou15";
			dim = 15;
			break;
		}
		else if (str == "1") {
			instance = "had20";
			dim = 20;
			break;
		}
		else if (str == "2") {
			instance = "lipa20a";
			dim = 20;
			break;
		}
		else if (str == "3") {
			instance = "nug30";
			dim = 30;
			break;
		}
		else if (str == "4") {
			instance = "tai25a";
			dim = 25;
			break;
		}
		else if (str == "5") {
			instance = "tai30a";
			dim = 30;
			break;
		}
		else if (str == "6") {
			instance = "tai30b";
			dim = 30;
			break;
		}
		else if (str == "7") {
			instance = "tai40b";
			dim = 40;
			break;
		}
		else if (str == "8") {
			instance = "tai80a";
			dim = 80;
			break;
		}
		else if (str == "9") {
			instance = "tai80b";
			dim = 80;
			break;
		}
		else if (str == "q") {
			exit(1);
		}
		else {
			std::cout << "Invalid value! Please try again.\n";
		}
	}
}

void SA::display_progress(const int& trial, const int& num_trial)
{
	double progress = (double)trial / (double)num_trial;
	int barWidth = 50;
	std::cout << "Progress: [";
	int pos = barWidth * progress;
	for (int i = 0; i < barWidth; i++) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << std::fixed << std::setprecision(1) << progress * 100 << " %\r" << std::flush;
}

std::string SA::getTimeStr()
{
	auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);
	struct tm tstruct {};
	localtime_s(&tstruct, &in_time_t);
	std::stringstream tmstr;
	tmstr << std::put_time(&tstruct, "%Y-%m-%d %X");
	return tmstr.str();
}

std::string SA::getDateStr()
{
	auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);
	struct tm tstruct {};
	localtime_s(&tstruct, &in_time_t);
	std::stringstream tmstr;
	tmstr << std::put_time(&tstruct, "%Y-%m-%d");
	return tmstr.str();
}

void PrintErrorMsg()
{
	std::cout << "Error: Invalid command line arguments\n";
	std::cout << "Usage: Run program in command line with 3 parameters. Order does not matter.\n";
	std::cout << "Example: ./name_of_program.exe --instance=0 --trial=30 --state=12\n";
	std::cout << "Avialable instances: (0) nug16a (1) nug30 (2) tai25a (3) tai50b\n";
	std::cin.get();
}

int ParsingArg(int argc, char* argv[], int& instance_num, int& num_trial, int& N_state)
{
	if (argc < 2)
	{
		PrintErrorMsg();
		return -1;
	}
	std::regex pattern("--(.*)=(.*)");

	try
	{
		for (size_t i = 1; i < argc; i++)
		{
			std::string param_to_search = argv[i];
			std::smatch match;
			if (std::regex_match(param_to_search, match, pattern))
			{
				auto it = match.begin();
				it++;

				if ((*it) == "instance")
				{
					it++;
					instance_num = std::stoi(*it);
					continue;
				}

				if ((*it) == "trial")
				{
					it++;
					num_trial = std::stoi(*it);
					continue;
				}

				if ((*it) == "state")
				{
					it++;
					N_state = std::stoi(*it);
					continue;
				}

				PrintErrorMsg();
				return -1;
			}
		}
	}
	catch (const std::exception& e)
	{
		std::cout << "Exception!\n";
		PrintErrorMsg();
		return -1;
	}
}