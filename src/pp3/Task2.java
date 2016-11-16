package pp3;

import java.util.ArrayList;
import java.util.List;
import Jama.Matrix;

/**
 * the class implements experiments in task 2
 * @author zhaokunxue
 *
 */
public class Task2 {
	
	/**
	 * get all updates ws
	 * @param training_data_with_labels training data set with labels
	 * @param testing_data_with_labels testing data set with labels
	 * @param filename data file name
	 * @param partial whether take partial records for gradient method
	 * @throws Exception
	 */
	public static void getWsAndPredictions(double[][] training_data_with_labels, 
			double[][] testing_data_with_labels, String filename, int partial) throws Exception {
		double[][] testing_data = ClassificationAlg.getDataFromDataWithLabels(testing_data_with_labels);
		double[] testing_labels = ClassificationAlg.getLabelsFromDataWithLabels(testing_data_with_labels);

		double[][] training_data_with_w0 = ClassificationAlg.addW0ToData(ClassificationAlg.getDataFromDataWithLabels(training_data_with_labels));
		double[] training_labels = ClassificationAlg.getLabelsFromDataWithLabels(training_data_with_labels);
		Matrix t_matrix = new Matrix(training_labels, 1).transpose();
		
		List<List<Double>> newton_times = new ArrayList<List<Double>>();
		List<List<Double>> gradient_times = new ArrayList<List<Double>>();
		List<Double> newton_avg_times = new ArrayList<Double>();
		List<Double> gradient_avg_times = new ArrayList<Double>();
		List<Matrix> newton_ws = new ArrayList<Matrix>();
		List<Matrix> gradient_ws = new ArrayList<Matrix>();
		List<Double> newton_error_rates = new ArrayList<Double>();
		List<Double> gradient_error_rates = new ArrayList<Double>();
		System.out.println(">>> Working on learning process <<<");
		for (int i = 0; i < 3; i++) {
			List<Double> newton_update_time = new ArrayList<Double>();
			List<Double> gradient_update_time = new ArrayList<Double>();
			newton_ws = DiscriminativeAlg.getUpdateWs(newton_update_time, training_data_with_w0, t_matrix, 0, partial);
			gradient_ws = DiscriminativeAlg.getUpdateWs(gradient_update_time, training_data_with_w0, t_matrix, 1, partial);	
			newton_times.add(newton_update_time);
			gradient_times.add(gradient_update_time);
		}
		
		newton_avg_times = PerformanceStat.calMeanTime(newton_times);
		gradient_avg_times = PerformanceStat.calMeanTime(gradient_times);
		System.out.println(">>> Predicting with newton method <<<");
		newton_error_rates = predictWithWs(newton_ws, training_data_with_w0, testing_data, testing_labels);
		String newton_output = "results/" + filename + "-task2-newton.csv";
		ClassificationAlg.writeTask2ToFile(newton_avg_times, newton_error_rates, newton_output);
		System.out.println(">>> Predicting with gradient method <<<");
		gradient_error_rates = predictWithWs(gradient_ws, training_data_with_w0, testing_data, testing_labels);
		String gradient_output = "results/" + filename + "-task2-gradient.csv";
		ClassificationAlg.writeTask2ToFile(gradient_avg_times, gradient_error_rates, gradient_output);
	}
	
	/**
	 * make predictions using ws we got
	 * @param ws a list of updated ws
	 * @param training_data_with_w0 training data with w0 added
	 * @param testing_data testing data set
	 * @param testing_labels testing data labels
	 * @return a list of error rates
	 */
	public static List<Double> predictWithWs(List<Matrix> ws, double[][] training_data_with_w0, 
			double[][] testing_data, double[] testing_labels) {
		List<Double> error_rates = new ArrayList<Double>();
		for (Matrix w : ws) {
			double error_rate = DiscriminativeAlg.predictForDifferentW(w, training_data_with_w0, testing_data, testing_labels);
			error_rates.add(error_rate);
		}
		return error_rates;
	}
	
	/**
	 * run the experiments for task2
	 * @throws Exception
	 */
	public static void runTask2() throws Exception {
		String dataPath = "data/";
		String[] filenames = new String[] {"A", "usps"};
		for (String filename : filenames) {
			String data_file = dataPath + filename + ".csv";
			String data_labels = dataPath + "labels-" + filename +".csv";
			System.out.println("---- Working on datafile: " + data_file + " ----");
			double[][] data_with_labels = ClassificationAlg.combineData(data_file, data_labels);
			int data_size = data_with_labels.length;
			int testing_size = data_size / 3;
			List<double[][]> splitted_data = ClassificationAlg.splitTestAndTrain(data_with_labels, testing_size);
			double[][] training_data_with_labels = splitted_data.get(0);
			double[][] testing_data_with_labels = splitted_data.get(1);
			int partial = 0;
			if (filename == "usps"){
				partial = 1;
			}
			getWsAndPredictions(training_data_with_labels, testing_data_with_labels, filename, partial);
		}

	}
}
