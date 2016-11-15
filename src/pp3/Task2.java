package pp3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import Jama.Matrix;

public class Task2 {
	public static void getWs(double[][] training_data_with_labels) {
		List<Matrix> newton_ws = new ArrayList<Matrix>();
		List<Double> newton_update_time = new ArrayList<Double>();
		List<Matrix> gradient_ws = new ArrayList<Matrix>();
		List<Double> gradient_update_time = new ArrayList<Double>();
		double[][] training_data = ClassificationAlg.getDataFromDataWithLabels(training_data_with_labels);
		//double[][] current_training_data = DiscriminativeAlg.addW0ToData(ClassificationAlg.getDataFromDataWithLabels(training_data_with_labels));
		double[] training_labels = ClassificationAlg.getLabelsFromDataWithLabels(training_data_with_labels);
		Matrix t_matrix = new Matrix(training_labels, 1).transpose();
		//DiscriminativeAlg.getUpdateWs(newton_ws, newton_update_time, current_training_data, current_t_matrix, 0);
		DiscriminativeAlg.getUpdateWs(gradient_ws, gradient_update_time, training_data, t_matrix, 1);
//		for (int i = 0; i < gradient_ws.size(); i++) {
//			System.out.println(gradient_update_time.get(i));
//			System.out.println(Arrays.deepToString(gradient_ws.get(i).getArray()));
//		}
//
//		System.out.println("time: " + gradient_update_time.size() + " ws: " + gradient_ws.size());
	}
	
//	public static void predictWithDiffUpdate(double[][] data_with_labels){
//		int data_size = data_with_labels.length;
//		int testing_size = data_size / 3;
//		int training_size = data_size - testing_size;
//		List<double[][]> splitted_data = ClassificationAlg.splitTestAndTrain(data_with_labels, testing_size);
//		double[][] training_data_with_labels = splitted_data.get(0);
//		double[][] testing_data_with_labels = splitted_data.get(1);
//		double[][] testing_data = ClassificationAlg.getDataFromDataWithLabels(testing_data_with_labels);
//		double[] testing_labels = ClassificationAlg.getLabelsFromDataWithLabels(testing_data_with_labels);
//		List<List<Double>> newton_data = new ArrayList<List<Double>>();
//		List<List<Double>> gradient_data = new ArrayList<List<Double>>();
//		for (int i = 0; i < 3; i++) {
//			//DiscriminativeAlg.diffUpdateMethodPredict(newton_data, training_data_with_labels, testing_data, testing_labels, training_size, 0);
//			DiscriminativeAlg.diffUpdateMethodPredict(gradient_data, training_data_with_labels, testing_data, testing_labels, training_size, 1);
//		}
////		for (List<Double> stat : gradient_data) {
////			for (double d : stat) {
////				System.out.print(d + " ");
////			}
////			System.out.println("");
////		}
//	}
	
	public static void runTask2() throws Exception {
		String dataPath = "data/";
		String dataA = dataPath + "A.csv";
		String dataA_labels = dataPath + "labels-A.csv";
		double[][] dataA_with_labels = ClassificationAlg.combineData(dataA, dataA_labels);
		int data_size = dataA_with_labels.length;
		int testing_size = data_size / 3;
		int training_size = data_size - testing_size;
		System.out.println(testing_size + ":" + training_size);
		List<double[][]> splitted_data = ClassificationAlg.splitTestAndTrain(dataA_with_labels, testing_size);
		double[][] training_data_with_labels = splitted_data.get(0);
		double[][] testing_data_with_labels = splitted_data.get(1);
		double[][] testing_data = ClassificationAlg.getDataFromDataWithLabels(testing_data_with_labels);
		double[] testing_labels = ClassificationAlg.getLabelsFromDataWithLabels(testing_data_with_labels);
		//predictWithDiffUpdate(dataA_with_labels);
		//System.out.println(Arrays.deepToString(training_data_with_labels));
		getWs(training_data_with_labels);
	}
}
