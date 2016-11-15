package pp3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import Jama.Matrix;

public class Task2 {
	public static void getWs(double[][] training_data_with_labels, 
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
		
		for (int i = 0; i < 3; i++) {
			List<Double> newton_update_time = new ArrayList<Double>();
			List<Double> gradient_update_time = new ArrayList<Double>();
			newton_ws = DiscriminativeAlg.getUpdateWs(newton_update_time, training_data_with_w0, t_matrix, 0, partial);
			gradient_ws = DiscriminativeAlg.getUpdateWs(gradient_update_time, training_data_with_w0, t_matrix, 1, partial);	
			newton_times.add(newton_update_time);
			gradient_times.add(gradient_update_time);
		}
		
		System.out.println("newton: " + newton_ws.size() + " : " + newton_times.get(0).size() + " : " + newton_times.get(1).size() + " : " + newton_times.get(2).size());
		System.out.println("gradient: " + gradient_ws.size() + " : " + gradient_times.get(0).size() + " : " + gradient_times.get(1).size() + " : " + gradient_times.get(2).size());
		newton_avg_times = PerformanceStat.calMeanTime(newton_times);
		gradient_avg_times = PerformanceStat.calMeanTime(gradient_times);
		for (Double d : gradient_avg_times){
			System.out.print(d + ", ");
		}
		System.out.println("");
		System.out.println(Arrays.deepToString((gradient_ws.get(gradient_ws.size()-1)).getArray()));
		newton_error_rates = predictWithWs(newton_ws, training_data_with_w0, testing_data, testing_labels);
		gradient_error_rates = predictWithWs(gradient_ws, training_data_with_w0, testing_data, testing_labels);
		System.out.println("error rates for newton w...");
		for (Double er : newton_error_rates) {
			System.out.print(er + ", ");
		}
		System.out.println(" \n error rates for gradient w...");
		for (Double er : gradient_error_rates) {
			System.out.print(er + ", ");
		}
		System.out.println("");
		String newton_output = "results/" + filename + "-task2-newton.csv";
		String gradient_output = "results/" + filename + "-task2-graident.csv";
		ClassificationAlg.writeTask2ToFile(newton_avg_times, newton_error_rates, newton_output);
		ClassificationAlg.writeTask2ToFile(gradient_avg_times, gradient_error_rates, gradient_output);
	}
	
	public static List<Double> predictWithWs(List<Matrix> ws, double[][] training_data_with_w0, 
			double[][] testing_data, double[] testing_labels) {
		List<Double> error_rates = new ArrayList<Double>();
		for (Matrix w : ws) {
			double error_rate = DiscriminativeAlg.predictForDifferentW(w, training_data_with_w0, testing_data, testing_labels);
			error_rates.add(error_rate);
		}
		return error_rates;
	}
	
	public static void runTask2() throws Exception {
		String dataPath = "data/";
		String[] filenames = new String[] {"A", "usps"};
		for (String filename : filenames) {
			String data_file = dataPath + filename + ".csv";
			String data_labels = dataPath + "labels-" + filename +".csv";
			double[][] data_with_labels = ClassificationAlg.combineData(data_file, data_labels);
			int data_size = data_with_labels.length;
			int testing_size = data_size / 3;
			int training_size = data_size - testing_size;
			System.out.println(testing_size + ":" + training_size);
			List<double[][]> splitted_data = ClassificationAlg.splitTestAndTrain(data_with_labels, testing_size);
			double[][] training_data_with_labels = splitted_data.get(0);
			double[][] testing_data_with_labels = splitted_data.get(1);
			//predictWithDiffUpdate(dataA_with_labels);
			//System.out.println(Arrays.deepToString(training_data_with_labels));
			int partial = 0;
			if (filename == "usps"){
				partial = 1;
			}
			getWs(training_data_with_labels, testing_data_with_labels, filename, partial);
		}

	}
}
