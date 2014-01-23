package net.rendicahya.algorithm;

import javax.swing.JLabel;

/**
 * Implementasi algoritma Jaringan Saraf Tiruan dengan metode pembelajaran
 * Backpropagation.
 *
 * @author Rendicahya
 */
public class Backpropagation {

    private double[][][] weight;
    private double[][] trainingSet;
    private double[][] neuron;
    private double[][] delta;
    private double[][] target;
    private double maxError = .01;
    private double learningRate = .1;
    private int maxEpochs = -1;
    private JLabel lblEpoch = null;
    private JLabel lblError = null;
    private double finalMse;

    public void setEpochLabel(JLabel lblEpoch) {
        this.lblEpoch = lblEpoch;
    }

    public void setErrorLabel(JLabel lblError) {
        this.lblError = lblError;
    }

    /**
     * Menerima konfigurasi sekaligus membentuk jaringan. Parameter berupa array
     * yang merepresentasikan jumlah layer dan jumlah neuron pada masing2 layer.
     *
     * Contoh: Untuk membentuk jaringan dengan 2 hidden layer dengan
     * konfigurasi: <ul> <li>3 neuron input</li> <li>4 neuron pada hidden layer
     * 1</li> <li>5 neuron pada hidden layer 2</li> <li>1 neuron output</li>
     * </ul>
     *
     * maka parameter yang dikirimkan adalah
     * <pre>
     * int[] konfigurasi = {3, 4, 5, 1};
     * buildNetwork(konfigurasi);
     * </pre>
     *
     * atau dengan bentuk parameter seperti berikut
     * <pre>
     * buildNetwork(3, 4, 5, 1);
     * </pre>
     *
     * @param layers konfigurasi jumlah layer dan neuron
     */
    public void buildNetwork(int... layers) {
        /*
         * init neurons
         */
        neuron = new double[layers.length][];
        delta = new double[layers.length][];
        final int layersNum = layers.length;

        /*
         * looping setiap lapisan
         */
        for (int layer = 0; layer < layersNum; layer++) {
            /*
             * untuk layer output, hanya ada neuron.
             * untuk layer selain layer output, ditambah bias.
             */
            neuron[layer] = layer == layersNum - 1 ? new double[layers[layer]] : new double[layers[layer] + 1];

            /*
             * delta error hanya ada pada neuron
             * (tidak pada bias)
             */
            delta[layer] = new double[layers[layer]];
        }

        /*
         * set nilai bias = 1
         */
        for (int layer = 0; layer < layersNum - 1; layer++) {
            neuron[layer][layers[layer]] = 1;
        }

        /*
         * init bobot
         */
        weight = new double[layersNum - 1][][];

        for (int layer = 0; layer < weight.length; layer++) {
            weight[layer] = new double[layers[layer] + 1][];

            for (int node = 0; node < weight[layer].length; node++) {
                weight[layer][node] = new double[layers[layer + 1]];
            }
        }
    }

    /**
     * Memasukkan bilangan acak ke semua bobot pada jaringan. Method ini
     * dipanggil setelah method
     * <code>buildNetwork()</code> dipanggil.
     */
    public void randomizeWeights() {
        for (int layer = 0; layer < weight.length; layer++) {
            for (int node = 0; node < weight[layer].length; node++) {
                for (int branch = 0; branch < weight[layer][node].length; branch++) {
                    weight[layer][node][branch] = -1 + Math.random() * 2;
                }
            }
        }
    }

    public void setWeights(double[] w) {
        for (int layer = 0, i = 0; layer < weight.length; layer++) {
            for (int node = 0; node < weight[layer].length; node++) {
                for (int branch = 0; branch < weight[layer][node].length; branch++, i++) {
                    weight[layer][node][branch] = w[i];
                }
            }
        }
    }

    public void setWeights(double[][][] w) {
        for (int layer = 0; layer < weight.length; layer++) {
            for (int node = 0; node < weight[layer].length; node++) {
                System.arraycopy(w[layer][node], 0, weight[layer][node], 0, weight[layer][node].length);
            }
        }
    }

    public void setTrainingData(double[][] trainingSet) {
        this.trainingSet = trainingSet;
    }

    /**
     * Mengeset target pembelajaran.
     *
     * @param target array yang berisi nilai-nilai target.
     */
    public void setTarget(double[][] target) {
        this.target = target;
    }

    /**
     * Mengeset error maksimum sehingga pembelajaran akan terus dilakukan selama
     * error masih di atas error maksimum.
     *
     * @param maxError nilai error maksimum.
     */
    public void setMaxError(double maxError) {
        this.maxError = maxError;
    }

    /**
     * Mengeset laju pembelajaran (learning rate).
     *
     * @param learningRate nilai laju pembelajaran.
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Mengeset epoh maksimum (maximum epoch) untuk proses pembelajaran. Untuk
     * epoh maksimum tak terhingga masukkan parameter dengan nilai
     * <code>-1</code>.
     *
     * @param maxEpochs
     */
    public void setMaxEpochs(int maxEpochs) {
        this.maxEpochs = maxEpochs;
    }

    /**
     * Memulai proses pembelajaran.
     */
    public void train() {
        final int layersNum = neuron.length;
        int epoch = 0;
        double mse = 0;

        /*
         * satu iterasi do-while adalah satu epoch
         * satu epoch adalah satu kali proses semua training set
         */
        do {
            double errorSum = 0;
            /*
             * looping setiap data training
             */
            int trainingSetLength = trainingSet.length;

            for (int set = 0; set < trainingSetLength; set++) {
                /*
                 * set data training ke lapisan masukan
                 */
                System.arraycopy(trainingSet[set], 0, neuron[0], 0, neuron[0].length - 1);

                /*
                 * feedforward
                 * looping untuk semua layer kecuali input layer
                 */
                for (int layer = 1; layer < layersNum; layer++) {
                    /*
                     * jika berada pada layer terakhir (layer output),
                     * maka semua neuron menerima input (karena tidak memiliki bias)
                     *
                     * jika berada pada layer selain layer output,
                     * maka ada 1 neuron yg tidak menerima input, yaitu bias
                     */
                    final int neuronsNum = layer == layersNum - 1 ? neuron[layer].length : neuron[layer].length - 1;

                    for (int node = 0; node < neuronsNum; node++) {
                        double sum = 0;

                        /*
                         * menjumlahkan output dari neuron2 yg berhubungan
                         * pada layer sebelumnya
                         */
                        int prevLayerLength = neuron[layer - 1].length;

                        for (int prevNode = 0; prevNode < prevLayerLength; prevNode++) {
                            sum += neuron[layer - 1][prevNode] * weight[layer - 1][prevNode][node];
                        }

                        /*
                         * menghitung output dari neuron
                         * dengan fungsi sigmoid
                         */
                        neuron[layer][node] = 1 / (1 + Math.exp(-sum));
                    }
                }

                /*
                 * hitung error d pada lapisan keluaran
                 */
                int outputLayerLength = neuron[layersNum - 1].length;

                for (int node = 0; node < outputLayerLength; node++) {
                    double output = neuron[layersNum - 1][node];
                    double diff = target[set][node] - output;
                    delta[layersNum - 1][node] = diff * output * (1 - output);
                    errorSum += diff * diff;
                }

                /*
                 * hitung error d
                 * dimulai dari lapisan tersembunyi yg terakhir
                 * (error d untuk lapisan keluaran sudah dikalkulasi di atas).
                 * lapisan masukan tidak memerlukan error d.
                 */
                for (int layer = layersNum - 2; layer > 0; layer--) {
                    int nodes = neuron[layer].length - 1;

                    for (int node = 0; node < nodes; node++) {
                        int branches = weight[layer][node].length;
                        double sum = 0;

                        for (int branch = 0; branch < branches; branch++) {
                            sum += delta[layer + 1][branch] * weight[layer][node][branch];
                        }

                        double y = neuron[layer][node];
                        delta[layer][node] = sum * y * (1 - y);
                    }
                }

                /*
                 * update bobot
                 */
                final int weightLayers = weight.length;

                for (int layer = 0; layer < weightLayers; layer++) {
                    final int nodes = weight[layer].length;

                    for (int node = 0; node < nodes; node++) {
                        final int branches = weight[layer][node].length;

                        for (int branch = 0; branch < branches; branch++) {
                            weight[layer][node][branch] += learningRate * delta[layer + 1][branch] * neuron[layer][node];
                        }
                    }
                }
            }

            mse = errorSum / 2;

            if (lblEpoch != null) {
                lblEpoch.setText(String.valueOf(epoch));
            }

            if (lblError != null) {
                lblError.setText(String.valueOf(mse));
            }
        } while (mse > maxError && (maxEpochs == -1 | ++epoch < maxEpochs));

        finalMse = mse;
    }

    /**
     * Memproses masukan dengan jaringan saraf tiruan yang yang telah dilatih.
     *
     * @param input array berupa
     * @return output dari jaringan saraf tiruan
     */
    public double[] test(double... input) {
        final int layersNum = neuron.length;

        /*
         * set data masukan ke lapisan masukan
         */
        System.arraycopy(input, 0, neuron[0], 0, neuron[0].length - 1);

        /*
         * feedforward only
         */
        for (int layer = 1; layer < layersNum; layer++) {
            final int neuronsNum = layer == layersNum - 1 ? neuron[layer].length : neuron[layer].length - 1;

            for (int node = 0; node < neuronsNum; node++) {
                double sum = 0;
                int prevLayerLength = neuron[layer - 1].length;

                for (int prevNode = 0; prevNode < prevLayerLength; prevNode++) {
                    sum += neuron[layer - 1][prevNode] * weight[layer - 1][prevNode][node];
                }

                neuron[layer][node] = 1 / (1 + Math.exp(-sum));
            }
        }

        return neuron[neuron.length - 1];
    }

//    public static void main(String[] args) {
//        Backpropagation bp = new Backpropagation();
//
////        file: BP test.ods
////        bp.buildNetwork(3, 2, 2);
////        bp.setWeights(new double[][][]{{{.123, .234}, {.345, .456}, {.567, .678}, {.789, .899}}, {{.123, .234}, {.234, .345}, {.345, .456}}});
////        bp.setTrainingData(new double[][]{{.1, .2, .3}});
////        bp.setTarget(new double[][]{{.75, .5}});
////        bp.setMaxEpochs(1);
////        bp.setLearningRate(.15);
//
//        bp.buildNetwork(8, 8, 1);
//        bp.randomizeWeights();
//        bp.setTrainingData(new double[][]{
//                    {94, 94, 78, 94, 109, 187, 110, 109},
//                    {93, 125, 94, 63, 140, 203, 157, 109},
//                    {78, 109, 94, 62, 141, 188, 140, 125},
//                    {78, 125, 62, 94, 125, 187, 125, 110},
//                    {110, 94, 93, 78, 141, 172, 125, 141},
//                    {344, 343, 313, 312, 360, 297, 343, 282},
//                    {297, 266, 312, 282, 281, 281, 313, 265},
//                    {235, 281, 297, 250, 328, 266, 281, 297},
//                    {234, 266, 266, 234, 281, 266, 312, 235},
//                    {219, 265, 235, 250, 312, 266, 297, 250}
//                });
//        bp.setTarget(new double[][]{{0}, {0}, {0}, {1}, {0}, {1}, {1}, {1}, {1}, {1},});
//        bp.setMaxEpochs(50000);
//        bp.setLearningRate(.1);
//
//        bp.train();
//        System.out.println(bp.test(94, 94, 78, 94, 109, 187, 110, 109)[0]);
//        System.out.println(bp.test(297, 266, 312, 282, 281, 281, 313, 265)[0]);
//    }
    public double getMse() {
        return finalMse;
    }
}