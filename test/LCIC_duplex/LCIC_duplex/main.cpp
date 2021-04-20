#include "arithmetic_codec.h"
#include "encode.h"
#include "decode.h"
#include "stdio.h"
#include <time.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <windows.h>


typedef std::vector<std::string> stringvec;

void read_directory(const std::string& name, stringvec& v)
{
	std::string pattern(name);
	pattern.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			v.push_back(data.cFileName);
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}
}

void printUsage(char* s) {
	printf("Usage: %s e [source file (ppm)] [compressed file (bin)]\n", s);
	printf("Usage: %s d [compressed file (bin)] [decoded file (bmp)]\n", s);
}

int main(int argc, char* argv[]) {

	bool ENCODE = true;
	bool DECODE = true;

	if (ENCODE) {
		std::string dir_name;

		dir_name = "../data/";

		stringvec v;

		read_directory(dir_name, v);
		v.erase(v.begin(), v.begin() + 2);

		int num_files = v.size();
		float avg_bpp = 0;

		std::string infile;
		struct stat st;
		float cur_bpp;

		char weights_smooth_y[40], weights_smooth_u[40], weights_smooth_v[40];
		char weights_texture_y[40], weights_texture_u[40], weights_texture_v[40];
		strcpy(weights_smooth_y, "../weights_smooth_y.txt");
		strcpy(weights_smooth_u, "../weights_smooth_u.txt");
		strcpy(weights_smooth_v, "../weights_smooth_v.txt");
		strcpy(weights_texture_y, "../weights_texture_y.txt");
		strcpy(weights_texture_u, "../weights_texture_u.txt");
		strcpy(weights_texture_v, "../weights_texture_v.txt");

		std::string codename = "code.bin";
		std::string code(".bin");
		std::string code_y, code_u, code_v;
		size_t pos = codename.find(code);
		code_y = codename.replace(pos, codename.length(), "_y.bin");
		code_u = codename.replace(pos, codename.length(), "_u.bin");
		code_v = codename.replace(pos, codename.length(), "_v.bin");

		char* chr_y = _strdup(code_y.c_str());
		char* chr_u = _strdup(code_u.c_str());
		char* chr_v = _strdup(code_v.c_str());

		float bpp;
		char* infilename;
		clock_t start, end;
		float avg_time = 0;


		for (int i = 0; i < num_files; i++) {
			infile = dir_name + v.at(i);
			infilename = &infile[0];

			start = clock();

			bpp = runEncoder(infilename, chr_y, chr_u, chr_v, weights_smooth_y, weights_smooth_u, weights_smooth_v, weights_texture_y, weights_texture_u, weights_texture_v);

			end = clock();

			avg_bpp += bpp;
			avg_time += (end - start) / (double)CLOCKS_PER_SEC;

			std::cout << "Current time : " << (end - start) / (double)CLOCKS_PER_SEC << std::endl;
		}

		avg_bpp = avg_bpp / num_files;
		avg_time = avg_time / num_files;

		std::cout << "Average BPP : " << avg_bpp << std::endl;
		std::cout << "Average time : " << avg_time << std::endl;

	}

	if (DECODE) {
		char weights_smooth_y[40], weights_smooth_u[40], weights_smooth_v[40];
		char weights_texture_y[40], weights_texture_u[40], weights_texture_v[40];
		strcpy(weights_smooth_y, "../weights_smooth_y.txt");
		strcpy(weights_smooth_u, "../weights_smooth_u.txt");
		strcpy(weights_smooth_v, "../weights_smooth_v.txt");
		strcpy(weights_texture_y, "../weights_texture_y.txt");
		strcpy(weights_texture_u, "../weights_texture_u.txt");
		strcpy(weights_texture_v, "../weights_texture_v.txt");

		std::string codename = "code.bin";
		std::string code(".bin");
		std::string code_y, code_u, code_v;
		size_t pos = codename.find(code);
		code_y = codename.replace(pos, codename.length(), "_y.bin");
		code_u = codename.replace(pos, codename.length(), "_u.bin");
		code_v = codename.replace(pos, codename.length(), "_v.bin");

		std::string outname = "out.ppm";

		char* chr_y = _strdup(code_y.c_str());
		char* chr_u = _strdup(code_u.c_str());
		char* chr_v = _strdup(code_v.c_str());

		char* out = _strdup(outname.c_str());

		runDecoder(chr_y, chr_u, chr_v, out, weights_smooth_y, weights_smooth_u, weights_smooth_v, weights_texture_y, weights_texture_u, weights_texture_v);
	}

	system("PAUSE");
}