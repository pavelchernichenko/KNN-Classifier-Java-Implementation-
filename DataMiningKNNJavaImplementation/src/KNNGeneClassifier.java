import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.*;

public class KNNGeneClassifier
{
    public static void main(String[] args)
    {
        try {
            GeneParserFromCSV parse = new GeneParserFromCSV();
            String[][] trainingArray = parse.getTrainingData();
            String[][] testArray = parse.getTestData();
            //include both data sets and select K value
            KNNClassifier knn = new KNNClassifier(trainingArray, testArray, 4);
            knn.predictionsAndOutput();
        } catch(Exception e){
            System.out.println("ERROR");
            e.printStackTrace();
        }
    }
}

final class GeneParserFromCSV
{
    private static String TEST_DATA_PATH = "test_data_post_rapid_miner.csv";
    private static String TRAINING_DATA_PATH = "training_data.csv";
    private static String[] GENE_ATTRIBUTES = {"GeneID", "Essential", "Class", "Complex", "Phenotype", "Motif", "Chromosome", "Localization"};
    private String[][] testData;
    private String[][] trainingData;
    private File testFile;
    private File trainingFile;
    private Scanner csvScanner;

    public GeneParserFromCSV() {
        try {
            this.testFile = new File(TEST_DATA_PATH);
            this.trainingFile = new File(TRAINING_DATA_PATH);
            Path testPath = Paths.get(TEST_DATA_PATH);
            Path trainingPath = Paths.get(TRAINING_DATA_PATH);
            int numLinesTestData = (int) Files.lines(testPath).count();
            int numLinesTrainingData = (int) Files.lines(trainingPath).count();
            this.testData = this.parseCSVFile(this.testFile, numLinesTestData);
            this.trainingData = this.parseCSVFile(this.trainingFile, numLinesTrainingData);

        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    private final String[][] parseCSVFile(final File gFile, int numLines) throws IOException
    {
        assert gFile != null;
        this.csvScanner = new Scanner(gFile);
        String currRow = "";
        int count = 0;
        String[] rowSplitArray = null;
        String[][] csvNewValues = new String[numLines][GENE_ATTRIBUTES.length];
        while ((currRow = csvScanner.nextLine()) != null && count < numLines - 1)
        {
            rowSplitArray = currRow.split(";");
            csvNewValues[count] = rowSplitArray;
            count++;
        }
        return csvNewValues;
    }

    public String[][] getTestData()
    {
        return testData;
    }
    public String[][] getTrainingData()
    {
        return trainingData;
    }

    public static void createOutputFile(final File resultsFile, String[] newData) throws IOException
    {
        assert resultsFile != null;
        PrintStream output = new PrintStream(resultsFile);
        output.println("<GENE ID>  <LOCALIZATION>");
        for (String str : newData) {
            output.println(str);
        }
        output.close();
    }
}

final class KNNClassifier
{
    private int TRAINING_INDEX_ID = 6;
    private int TEST_INDEX_ID = 7;
    private String[][] trainingData;
    private String[][] testData;
    private BuildNeighborhood neighbors;
    private HashMap<String, List<String>> kNearestNeighbors = new HashMap<>();
    private HashMap<String, String> localPredictions = new HashMap<>();
    private HashMap<String, String> localizationsOfficial;
    private int k;

    public KNNClassifier(String[][] trainingData, String[][] testData, int k)
    {
        this.trainingData = trainingData;
        this.testData = testData;
        this.k = k;
        this.kNearestNeighbors = getNearestNeighbors();
        predictJob();
    }

    public HashMap<String, List<String>> getNearestNeighbors()
    {
        this.neighbors = new BuildNeighborhood();
        this.neighbors.figureOutNeighborhood(this.trainingData, this.testData);

        HashMap<String, List<String>> kNbrs = new HashMap<>();
        HashMap<String, ArrayList<BuildNeighborhood.Pair<String, Integer>>> nearestNeighbors = this.neighbors.getNearestNeighbors();
        Set<String> set = nearestNeighbors.keySet();

        for(String key : set)
        {
            ArrayList<BuildNeighborhood.Pair<String, Integer>> list = nearestNeighbors.get(key);
            int[] maxValues = new int[this.k];
            String[] maxValueString = new String[this.k];

            for(BuildNeighborhood.Pair<String, Integer> pair : list)
            {
                for(int i = 0; i < maxValues.length; i++)
                {
                    if(pair.getElement1() > maxValues[i] && !Arrays.asList(maxValueString).contains(pair.getElement0()))
                    {
                        maxValues[i] = pair.getElement1();
                        maxValueString[i] = pair.getElement0();
                    }
                }
            }
            kNbrs.put(key, Arrays.asList(maxValueString));
        }
        return kNbrs;
    }
    public void predictJob()
    {
        HashMap<String, String> finLocalizations = new HashMap<>();
        HashMap<String, String> trainingLocalizations = new HashMap<>();

        for(String[] testGeneTuples : this.testData)
        {
            finLocalizations.put(testGeneTuples[TEST_INDEX_ID], testGeneTuples[TRAINING_INDEX_ID]);
        }
        for(String[] trainingGeneTuples : this.trainingData)
        {
            trainingLocalizations.put(trainingGeneTuples[TRAINING_INDEX_ID], trainingGeneTuples[TEST_INDEX_ID]);
        }
        for(String idVal : this.kNearestNeighbors.keySet())
        {
            List<String> kNeighbors = this.kNearestNeighbors.get(idVal);
            HashMap<String, Integer> knnWithWeightsHash = new HashMap<>();
            ArrayList<BuildNeighborhood.Pair<String, Integer>> allNeighbors = this.neighbors.getNearestNeighbors().get(idVal);

            for(BuildNeighborhood.Pair<String, Integer> neighbor : allNeighbors)
            {
                if(kNeighbors.contains(neighbor.getElement0()))
                {
                    knnWithWeightsHash.put(neighbor.getElement0(), neighbor.getElement1());
                }
            }
            String localizationPrediction = "";
            int maxVal = 0;
            for(Integer wVal : knnWithWeightsHash.values())
            {
                if(wVal > maxVal)
                {
                    maxVal = wVal;
                }
            }
            List<String> localizationsList = new ArrayList<>();
            for(String setVal : knnWithWeightsHash.keySet())
            {
                if(knnWithWeightsHash.get(setVal) >= maxVal)
                {
                    localizationsList.add(0, setVal);
                }
            }
            localizationPrediction = localizationsList.get(0);
            String predictionText = trainingLocalizations.get(localizationPrediction);
            localPredictions.put(idVal, predictionText);
        }
        localizationsOfficial = finLocalizations;
    }

    public void predictionsAndOutput() throws IOException
    {

        File outFile = new File("OutputResults.csv");
        List<String> resultsForOutput = new ArrayList<>();
        DecimalFormat df2 = new DecimalFormat("#");
        int total = localPredictions.size();
        int correctLocalizationPredictions = 0;
        System.out.println("\n<MODEL PREDICTION>\t\t\t\t\t\t<CORRECT PREDICTION>\n");

        for(String key : localPredictions.keySet())
        {
            String actualLocalization = localizationsOfficial.get(key);
            String predictedLocalization = localPredictions.get(key);
            System.out.println("ID " + key + ": " + localPredictions.get(key) + "\t\t\t\tCorrect: " + actualLocalization);
            resultsForOutput.add(key + " | " +  localPredictions.get(key));
            if(actualLocalization.equals(predictedLocalization))
            {
                correctLocalizationPredictions++;
            }
        }
        double accuracy =  ((double)correctLocalizationPredictions / (double)total) * 100;
        System.out.println("\nAccuracy: " + df2.format(accuracy) + "%");
        GeneParserFromCSV.createOutputFile(outFile, resultsForOutput.toArray(new String[resultsForOutput.size()]));
    }
}

final class BuildNeighborhood
{
    public static int testDataIDIndex = 7;
    public static int testDataClassIndex = 1;
    public static int testDataComplexIndex = 2;
    public static int testDataMotifIndex = 4;
    public static int trainingDataIDIndex = 6;
    public static int trainingDataClassIndex = 1;
    public static int trainingDataComplexIndex = 2;
    public static int trainingDataMotifIndex = 4;
    public static int COMPLEX_WEIGHT = 1000;
    public static int CLASS_WEIGHT= 100;
    public static int MOTIF_WEIGHT = 10;

    public HashMap<String, ArrayList<Pair<String, Integer>>> getNearestNeighbors()
    {
        return nearestNeighbors;
    }

    private HashMap<String, ArrayList<Pair<String, Integer>>> nearestNeighbors = new HashMap<>();

    public void figureOutNeighborhood(String[][] trainingData, String[][] testData)
    {
        List<String[]> array = Arrays.asList(testData);
        for(String[] testTuple : array)
        {
            if(testTuple[0] == null){
                continue;
            }
            ArrayList<Pair<String, Integer>> arr = new ArrayList<Pair<String, Integer>>();
            if(nearestNeighbors.containsKey(testTuple[testDataIDIndex])){

                continue;
            }
            nearestNeighbors.put(testTuple[testDataIDIndex], arr);
            for(String[] trainingTuple: trainingData)
            {
                //test all similarities and add weights accordingly depending on the tuples value
                Pair<String, Integer> trainingTuplePair = new Pair<>(trainingTuple[trainingDataIDIndex], 0);
                int weight = 0;
                try {
                    if (testTuple[testDataClassIndex].equals(trainingTuple[trainingDataClassIndex]))
                    {
                        weight += CLASS_WEIGHT;
                    }
                    if (testTuple[testDataComplexIndex].equals(trainingTuple[trainingDataComplexIndex]))
                    {
                        weight += COMPLEX_WEIGHT;
                    }
                    if (testTuple[testDataMotifIndex].equals(trainingTuple[trainingDataMotifIndex]))
                    {
                        weight += MOTIF_WEIGHT;
                    }
                } catch (Exception e){
                    e.printStackTrace();
                }
                trainingTuplePair.setElement1(weight);
                arr.add(trainingTuplePair);
            }
        }
    }

    final class Pair<K, V>
    {

        public void setElement0(K element0) {
            this.element0 = element0;
        }

        public void setElement1(V element1) {
            this.element1 = element1;
        }

        private K element0;
        private V element1;

        public <K, V> Pair<K, V> createPair(K element0, V element1) {
            return new Pair<K, V>(element0, element1);
        }

        public Pair(K element0, V element1) {
            this.element0 = element0;
            this.element1 = element1;
        }

        public K getElement0() {
            return element0;

        }

        public V getElement1() {
            return element1;
        }
    }
}

    
