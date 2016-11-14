package pp3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import Jama.Matrix;

public class Task1 {
	
	public static HashMap<Integer, List<Double>> predict(double[][] data_with_labels) { 
		int data_size = data_with_labels.length;
		int testing_size = data_size / 3;
		int training_size = data_size - testing_size;
		HashMap<Integer, List<Double>> predict_results = new HashMap<Integer, List<Double>>();
		for (int i = 0; i < 30; i++) {
			double[][] shuffled_data = ClassificationAlg.shuffleData(data_with_labels);
			List<double[][]> splitted_data = ClassificationAlg.splitTestAndTrain(shuffled_data, testing_size);
			double[][] training_data_with_labels = splitted_data.get(0);
			double[][] testing_data_with_labels = splitted_data.get(1);
			double[][] testing_data = ClassificationAlg.getDataFromDataWithLabels(testing_data_with_labels);
			double[] testing_labels = ClassificationAlg.getLabelsFromDataWithLabels(testing_data_with_labels);
			for (int k = 1; k < 11; k++) {
				int n = 0;
				if (k == 10) {
					n = training_size;
				} else {
					n = (training_size/10) * k;
				}

				double[][] current_training_data = ClassificationAlg.getFirstNData(training_data_with_labels, n);
				List<double[][]> data_by_class = GenerativeAlg.splitDataByClass(current_training_data);
				double[][] data_in_c1 = ClassificationAlg.getDataFromDataWithLabels(data_by_class.get(0));
				double[][] data_in_c2 = ClassificationAlg.getDataFromDataWithLabels(data_by_class.get(1));
				List<Double> ns = GenerativeAlg.countNs(data_by_class);
				List<Matrix> mus = GenerativeAlg.calculateMus(data_by_class);
				Matrix mu1 = mus.get(0);
				Matrix mu2 = mus.get(1);
				Matrix s1 = GenerativeAlg.calculateSForClass(data_in_c1, mu1);
				Matrix s2 = GenerativeAlg.calculateSForClass(data_in_c2, mu2);
				Matrix s = GenerativeAlg.calculateS (s1, s2, ns);
				double w0 =  GenerativeAlg.calculateW0 (mu1, mu2, s, ns);
				Matrix w = GenerativeAlg.calculateW (mu1, mu2, s);
				double errors = 0;
				for (int j = 0; j < testing_data.length; j++) {
					Matrix t_j = new Matrix(testing_data[j], 1).transpose();
					//System.out.println("t_j row: " + t_j.getRowDimension() + " t_j col: " + t_j.getColumnDimension());
					double a_value = w.transpose().times(t_j).get(0,0) + w0;
					int predict_class = ClassificationAlg.sigmoidPredict(a_value);
					int true_label = (int)testing_labels[j];
					if (predict_class != true_label) {
						errors++;
					}
				}

				double error_rate = errors / (double)(testing_labels.length);
				if (predict_results.containsKey(n)) {
					List<Double> predicts = predict_results.get(n);
					predicts.add(error_rate);
					predict_results.put(n, predicts);
				} else {
					List<Double> predicts = new ArrayList<Double>();
					predicts.add(error_rate);
					predict_results.put(n, predicts);
				}
			}
		}
		return predict_results;
	}

	
	public static HashMap<Integer, double[]> getStatics(HashMap<Integer, List<Double>> predicts_errors) {
		HashMap<Integer, double[]> error_statics = new HashMap<Integer, double[]>();
		for (Map.Entry<Integer, List<Double>> entry : predicts_errors.entrySet()) {
		    int key = entry.getKey();
		    List<Double> errors = entry.getValue();
		    double mean = ErrorStat.calcMean(errors); 
		    double std = ErrorStat.calStD(errors);
		    double[] statics = new double[2];
		    statics[0] = mean;
		    	statics[1] = std;
		    error_statics.put(key, statics);
		}
		return error_statics;
	}
	
	public static void runTask1 () throws Exception{
		String dataPath = "data/";
		String dataA = dataPath + "A.csv";
		String dataA_labels = dataPath + "labels-A.csv";
		double[][] dataA_with_labels = GenerativeAlg.readData(dataA, dataA_labels);
		HashMap<Integer, List<Double>> predicts = predict(dataA_with_labels);
		HashMap<Integer, double[]> predicts_stat = getStatics(predicts);
		String outputA = "results/predicts-A.csv";
		ClassificationAlg.writeDataToFile(predicts_stat, outputA);
		
//		System.out.println("##############");
//		for (Map.Entry<Integer, List<Double>> entry : predicts.entrySet()) {
//			int key = entry.getKey();
//		    List<Double> errors = entry.getValue();
//			System.out.println("key: " + key + errors.size() + " \n " + Arrays.deepToString(errors.toArray()));
//			System.out.println("##############");
//		}
	}
}
