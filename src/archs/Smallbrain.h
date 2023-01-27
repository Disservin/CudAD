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
#ifndef CUDAD_SRC_ARCHS_SMALLBRAIN_H_
#define CUDAD_SRC_ARCHS_SMALLBRAIN_H_

#include "../activations/Linear.h"
#include "../activations/ReLU.h"
#include "../activations/Sigmoid.h"
#include "../data/SArray.h"
#include "../data/SparseInput.h"
#include "../dataset/dataset.h"
#include "../layer/DenseLayer.h"
#include "../layer/DuplicateDenseLayer.h"
#include "../loss/Loss.h"
#include "../loss/MPE.h"
#include "../optimizer/Adam.h"
#include "../optimizer/Optimiser.h"
#include "../position/fenparsing.h"

#include <tuple>

class Smallbrain
{

  public:
    static constexpr int Buckets = 4;
    static constexpr int Features = 12 * 64;
    static constexpr int Inputs = Buckets * Features;
    static constexpr int L2 = 512;
    static constexpr int Outputs = 1;
    static constexpr float SigmoidScalar = 2.5 / 400;

    static Optimiser *get_optimiser()
    {
        Adam *optim = new Adam();
        optim->lr = 1e-2;
        optim->beta1 = 0.95;
        optim->beta2 = 0.999;
        return optim;
    }

    static Loss *get_loss_function()
    {
        MPE *loss_f = new MPE(2.5, false);

        return loss_f;
    }

    static std::vector<LayerInterface *> get_layers()
    {
        DuplicateDenseLayer<Inputs, L2, ReLU> *l1 = new DuplicateDenseLayer<Inputs, L2, ReLU>();
        l1->lasso_regularization = 1.0 / 8388608.0;

        DenseLayer<L2 * 2, Outputs, Sigmoid> *l2 = new DenseLayer<L2 * 2, Outputs, Sigmoid>();
        dynamic_cast<Sigmoid *>(l2->getActivationFunction())->scalar = SigmoidScalar;

        // /********
        // old training
        // *********/

        // auto *l1 = new DenseLayer<Inputs, L2, ReLU>();
        // auto *l2 = new DenseLayer<L2, Outputs, Sigmoid>();

        // /********
        // old quantisation
        // *********/
        // // auto *l1 = new DenseLayer<Inputs, L2, Linear>();
        // // auto *l2 = new DenseLayer<L2, Outputs, Sigmoid>();

        // dynamic_cast<Sigmoid *>(l2->getActivationFunction())->scalar = SigmoidScalar;

        return std::vector<LayerInterface *>{l1, l2};
    }

    static void assign_inputs_batch(DataSet &positions, SparseInput &in1, SparseInput &in2, SArray<float> &output,
                                    SArray<bool> &output_mask, int epoch)
    {

        ASSERT(positions.positions.size() == in1.n);
        ASSERT(positions.positions.size() == in2.n);

        in1.clear();
        in2.clear();
        output_mask.clear();

#pragma omp parallel for schedule(static) num_threads(8)
        for (int i = 0; i < positions.positions.size(); i++)
            assign_input(positions.positions[i], in1, in2, output, output_mask, i, epoch);
    }

    static int king_square_index(Square k_sq)
    {
        // clang-format off
        constexpr int KING_BUCKET[64]{
            0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 1, 1, 1, 1,
            2, 2, 2, 2, 3, 3, 3, 3,
            2, 2, 2, 2, 3, 3, 3, 3,
            2, 2, 2, 2, 3, 3, 3, 3,
            2, 2, 2, 2, 3, 3, 3, 3,
        };
        // clang-format on

        return KING_BUCKET[k_sq];
    }

    template <Color view> static int index(Square psq, Piece p, Square kingSquare)
    {
        /*relative*/
        if (view == BLACK)
        {
            psq = mirrorVertically(psq);
            kingSquare = mirrorVertically(kingSquare);
        }

        return king_square_index(kingSquare) * Features + psq + (getPieceType(p) + (getPieceColor(p) != view) * 6) * 64;
    }

    static void assign_input(Position &p, SparseInput &in1, SparseInput &in2, SArray<float> &output,
                             SArray<bool> &output_mask, int id, int epoch)
    {

        // track king squares
        Square wKingSq = p.getKingSquare<WHITE>();
        Square bKingSq = p.getKingSquare<BLACK>();

        BB bb{p.m_occupancy};
        int idx = 0;
        int count = bitCount(bb);

        auto view = p.m_meta.getActivePlayer();

        while (bb)
        {
            Square sq = bitscanForward(bb);
            Piece pc = p.m_pieces.getPiece(idx);

            auto inp_idx_w = index<WHITE>(sq, pc, wKingSq);
            auto inp_idx_b = index<BLACK>(sq, pc, bKingSq);

            if (view == WHITE)
            {
                in1.set(id, inp_idx_w);
                in2.set(id, inp_idx_b);
            }
            else
            {
                in2.set(id, inp_idx_w);
                in1.set(id, inp_idx_b);
            }

            bb = lsbReset(bb);
            idx++;
        }

        float p_value = p.m_result.score;
        float w_value = p.m_result.wdl;

        // flip if black is to move -> relative network style
        if (p.m_meta.getActivePlayer() == BLACK)
        {
            p_value = -p_value;
            w_value = -w_value;
        }

        float p_target = 1 / (1 + expf(-p_value * SigmoidScalar));
        float w_target = (w_value + 1) / 2.0f;

        static constexpr float start_lambda = 0.5;
        static constexpr float end_lambda = 0.5;

        static constexpr int lambda = start_lambda;

        output(id) = lambda * p_target + (1 - lambda) * w_target;

        output_mask(id) = true;
    }
};

#endif // CUDAD_SRC_ARCHS_SMALLBRAIN_H_
