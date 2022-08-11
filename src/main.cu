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

    const string data_path = "E:\\Github\\CudAD\\src\\resources\\data\\trained_txt\\";
    const string data_output_path = "E:\\Github\\CudAD\\src\\resources\\data\\";
    const string output = "E:\\Github\\CudAD\\src\\resources\\runs\\L2_256_4\\";

    /*

    convert fen to bin

     */
    // for (int i = 0; i <= 11; i++)
    // {
    //     std::string filename = data_path + "data" + to_string(i) + ".txt";
    //     DataSet ds = read<TEXT>(filename);
    //     write(data_output_path + "data" + to_string(i) + ".bin", ds);
    // }

    // Shuffle data
    //  vector<string> files{};
    //  for (int i = 0; i <= 11; i++)
    //  {
    //      files.push_back(data_output_path + "data" + to_string(i) + ".bin");
    //  }
    //  mix_and_shuffle(files, data_output_path, 3);

    // Training
    vector<string> files{};
    // for (int i = 1; i <= 6; i++)
    //     files.push_back(data_path + "shuffled_" + to_string(i) + ".bin");

    for (int i = 0; i <= 3; i++)
        files.push_back(data_output_path + "smallbrain_shuffle.d9." + to_string(i) + ".bin");

    // Trainer<Smallbrain> trainer{};
    // trainer.fit(files, vector<string>{data_output_path + "smallbrain_shuffle.d9.0.bin"}, output);

    auto layers = Smallbrain::get_layers();
    Network network{layers};
    network.loadWeights(output + "weights-epoch270.nnue");
    BatchLoader batch_loader{files, 16384};
    batch_loader.start();

    computeScalars<Smallbrain>(batch_loader, network, 1024);
    quantitize_shallow(output + "270.net", network, 32, 512);

    // close();
}
