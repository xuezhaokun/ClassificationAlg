package pp3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import Jama.Matrix;

public class Task1 {
	
	public static List<HashMap<Integer, List<Double>>> predict(double[][] data_with_labels) { 
		int data_size = data_with_labels.length;
		int testing_size = data_size / 3;
		int training_size = data_size - testing_size;
		HashMap<Integer, List<Double>> generative_predict_results = new HashMap<Integer, List<Double>>();
		HashMap<Integer, List<Double>> discriminative_predict_results = new HashMap<Integer, List<Double>>();
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
				double[][] current_training_data_with_labels = ClassificationAlg.getFirstNData(training_data_with_labels, n);
				GenerativeAlg.generativePredict(generative_predict_results, current_training_data_with_labels, testing_data, testing_labels, n);
				DiscriminativeAlg.discriminativePredict(discriminative_predict_results, current_training_data_with_labels, testing_data, testing_labels, n);
			}
		}
		List<HashMap<Integer, List<Double>>> predicts = new ArrayList<HashMap<Integer, List<Double>>>();
		predicts.add(generative_predict_results);
		predicts.add(discriminative_predict_results);
		return predicts;
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
		double[][] dataA_with_labels = ClassificationAlg.combineData(dataA, dataA_labels);
		HashMap<Integer, List<Double>> generative_predicts = predict(dataA_with_labels).get(0);
		HashMap<Integer, List<Double>> discriminative_predicts = predict(dataA_with_labels).get(1);
		HashMap<Integer, double[]> generative_predicts_stat = getStatics(generative_predicts);
		HashMap<Integer, double[]> discriminative_predicts_stat = getStatics(discriminative_predicts);
		String outputA = "results/generative-predicts-A.csv";
		String outputB = "results/discriminative-predicts-A.csv";
		ClassificationAlg.writeDataToFile(generative_predicts_stat, outputA);
		ClassificationAlg.writeDataToFile(discriminative_predicts_stat, outputB);
		
//		System.out.println("##############");
//		for (Map.Entry<Integer, List<Double>> entry : predicts.entrySet()) {
//			int key = entry.getKey();
//		    List<Double> errors = entry.getValue();
//			System.out.println("key: " + key + errors.size() + " \n " + Arrays.deepToString(errors.toArray()));
//			System.out.println("##############");
//		}
	}
}
