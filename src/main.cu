/**
    CudAD is a CUDA neural network trainer, specific for chess engines.
    Copyright (C) 2022 Finn Eggers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "archs/Smallbrain.h"
#include "dataset/shuffle.h"
#include "dataset/writer.h"
#include "misc/config.h"
#include "trainer.h"

#include <iostream>
#include <vector>

using namespace std;

int main()
{
    init();

    const string folder = "2022-12-25-depth9";

    // DATA
    const string data_path_txt = "H:\\data-generation\\" + folder + "\\data\\";
    const string data_path_bin = "H:\\data-generation\\" + folder + "\\data-bin\\";
    const string mixed = "H:\\data-generation\\" + folder + "\\mixed\\";

    // TRAINING
    const string outName = "L2_512_4_d9_800m_piececount_lambda_fixed_2";
    const string output = "H:\\CudAD\\src\\resources\\runs\\" + outName + "\\";
    const string validation = "H:\\CudAD\\src\\resources\\data\\verification\\smallbrain-verification-3.bin";

    int numFiles = 16;
    string name = "smallbrain-depth9-";
    // *************************
    // convert fen to bin
    // *************************

    // for (int i = 0; i <= 30; i++)
    // {
    //     std::string filename = data_path_txt + "data" + to_string(i) + ".txt";

    //     DataSet ds = read<TEXT>(filename);
    //     write(data_path_bin + "data" + to_string(i) + ".bin", ds);
    // }

    // *************************
    // Shuffle data
    // *************************

    // vector<string> files_bin{};

    // for (int i = 0; i <= 30; i++)
    // {
    //     files_bin.push_back(data_path_bin + "data" + to_string(i) + ".bin");
    // }

    // mix_and_shuffle_2(files_bin, mixed + name + "$.bin", numFiles);
    // files_bin.clear();

    // *************************
    // Validation
    // *************************

    vector<string> files_bin{};

    files_bin.push_back("H:\\data-generation\\2022-12-25-depth9\\mixed\\smallbrain-depth9-1.bin");

    mix_and_shuffle_2(files_bin, "H:\\CudAD\\src\\resources\\data\\verification\\" + name + "$.bin", 2);
    files_bin.clear();

    const string validation_name = "H:\\CudAD\\src\\resources\\data\\verification\\smallbrain-verification-1.bin";
    // *************************
    // Training
    // *************************

    vector<string> files{};

    for (int i = 1; i <= numFiles; i++)
        files.push_back(mixed + name + to_string(i) + ".bin");

    Trainer<Smallbrain> trainer{};
    trainer.fit(files, vector<string>{validation_name}, output);

    /********
    Quantize weights
    *********/
    // auto layers = Smallbrain::get_layers();
    // Network network{layers};

    // for (int i = 400; i <= 450; i += 10)
    // {
    //     network.loadWeights(output + "weights-epoch" + std::to_string(i) + ".nnue");
    //     std::string fen = "8/7k/8/6KP/2B4P/8/8/8 w - - 49 114";
    //     test_fen<Smallbrain>(network, fen);

    // BatchLoader batch_loader{files, 16384};
    // batch_loader.start();
    // computeScalars<Smallbrain>(batch_loader, network, 1024);
    // quantitize_shallow(output + "default" + std::to_string(i) + ".nnue", network, 16, 512);
    // }

    close();
}
