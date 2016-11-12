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
			Matrix testing_matrix = new Matrix(testing_data);
			
			for (int n = 20; n < training_size; n = n + 20) {
				
				if ((n + 20) > training_size) {
					n = training_size;
				}
				
				double[][] current_training_data = ClassificationAlg.getFirstNData(training_data_with_labels, n);
				List<double[][]> splitted_class_with_labels = GenerativeAlg.splitDataByClass(current_training_data);
				List<Double> ns = GenerativeAlg.countNs(splitted_class_with_labels);
				List<Double> mus = GenerativeAlg.calculateMus(splitted_class_with_labels);
				double mu1 = mus.get(0);
				double mu2 = mus.get(1);
				Matrix s = GenerativeAlg.calculateS(splitted_class_with_labels);
				double w0 = GenerativeAlg.calculateW0(s, mus, ns);
				int colDim = s.getColumnDimension();
				Matrix mu_diff = new Matrix(colDim, 1, (mu1 - mu2));
				Matrix w = s.inverse().times(mu_diff);
				double[] w_vector = w.getColumnPackedCopy();
				double errors = 0;
				
				int label_index = testing_data_with_labels[0].length;
				int dimension = testing_data_with_labels[0].length - 1;
				for (int j = 0; j < testing_data_with_labels.length; j++) {
					double a_value = 0;
					for (int m = 0; m < dimension; m++) {
						a_value += testing_data_with_labels[j][m] * w_vector[m];
					}
					a_value += w0;
					int predict_class = ClassificationAlg.sigmoidPredict(a_value);
					int true_label = (int)testing_data_with_labels[j][label_index - 1];
					if (predict_class != true_label) {
						errors++;
					}
				}
//				double[] as = testing_matrix.times(w).getRowPackedCopy();
//				double errors = 0;
//				for (int m = 0; m < testing_label.length; m++) {
//					double a_value = as[m] + w0;
//					double predict_class = ClassificationAlg.sigmoidPredict(a_value);
//					
//					if (predict_class != testing_label[m]) {
//						errors++;
//					}
//				}
				double error_rate = errors/(double)(testing_labels.length);
				if (n == 20) {
					System.out.println("20 \n" + errors);
				} else if (n == 134) {
					System.out.println("134 \n" + errors);
				}
				
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
//		for (Map.Entry<Integer, List<Double>> entry : predict_results.entrySet()) {
//			int key = entry.getKey();
//		    List<Double> errors = entry.getValue();
//			System.out.println("key: " + key + errors.size() + " \n " + Arrays.deepToString(errors.toArray()));
//			System.out.println("##############");
//		}
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
		String dataA = dataPath + "B.csv";
		String dataA_labels = dataPath + "labels-B.csv";
		double[][] dataA_with_labels = GenerativeAlg.readData(dataA, dataA_labels);
		HashMap<Integer, List<Double>> predicts = predict(dataA_with_labels);
		HashMap<Integer, double[]> predicts_stat = getStatics(predicts);
		String outputA = "results/predicts-B.csv";
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
