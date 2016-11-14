package pp3;

import java.util.HashMap;
import java.util.List;

public class Task2 {
	public static void predictWithDiffUpdate(double[][] data_with_labels){
		int data_size = data_with_labels.length;
		int testing_size = data_size / 3;
		int training_size = data_size - testing_size;
		List<double[][]> splitted_data = ClassificationAlg.splitTestAndTrain(data_with_labels, testing_size);
		double[][] training_data_with_labels = splitted_data.get(0);
		double[][] testing_data_with_labels = splitted_data.get(1);
		double[][] testing_data = ClassificationAlg.getDataFromDataWithLabels(testing_data_with_labels);
		double[] testing_labels = ClassificationAlg.getLabelsFromDataWithLabels(testing_data_with_labels);
		
		for (int i = 0; i < 3; i++) {
			DiscriminativeAlg.discriminativePredict(predict_results, training_data_with_labels, testing_data, testing_labels, training_size, 0);
			DiscriminativeAlg.discriminativePredict(predict_results, training_data_with_labels, testing_data, testing_labels, training_size, 1);
		}
	}
}
