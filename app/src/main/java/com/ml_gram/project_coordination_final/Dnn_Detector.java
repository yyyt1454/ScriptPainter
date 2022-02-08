package com.ml_gram.project_coordination_final;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.icu.util.Output;
import android.os.SystemClock;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

import javax.xml.transform.Result;

/**
 * DNN object detector
 * [jobs]
 * - Find accurate coordination of cursor in given bitmap image
 *
 * [inputs / outputs]
 * - input : Bitmap frame
 * - output : objectness, Rect coords, score
 *
 */
public class Dnn_Detector {
    private String TAG = "Dnn_Detector";
    private Context context;
    private Detector detector;
    private Output_Data output_data;

    private boolean TF_OD_CPU_OR_GPU = false;
    private static final int TF_OD_API_INPUT_SIZE = 640;
    private static final String TF_OD_API_LABELS_FILE = "coordination_labels.txt";
    private boolean TF_OD_API_IS_QUANTIZED = false;
    private String TF_OD_API_MODEL_FILE;
    private float MINIMUM_CONFIDENCE_TF_OD_API = 0.80f;

    // outputs
    public class Output_Data {
        public class recognitions {
            public Float confidence;
            public RectF location;
        }
        public ArrayList<recognitions> outputs;

        public Output_Data () {
             outputs = new ArrayList<>();
        }

        public void add(float confidence, RectF locations) {
            recognitions recogs = new recognitions();
            recogs.confidence = confidence;
            recogs.location = locations;
            this.outputs.add(recogs);
        }
    }


    public Dnn_Detector(Context context, Integer model_index, boolean cpu_or_gpu) {
        this.context = context;
        this.TF_OD_CPU_OR_GPU = cpu_or_gpu;
        this.output_data = new Output_Data();

        switch (model_index) {
            case 0 :
                TF_OD_API_IS_QUANTIZED = true;
                TF_OD_API_MODEL_FILE = "201223_640640_quantized_metaadded.tflite";
                break;
            case 1:
                TF_OD_API_IS_QUANTIZED = true;
                TF_OD_API_MODEL_FILE = "210324_640640_quantized.tflite";
                break;
            case 2:
                TF_OD_API_IS_QUANTIZED = true;
                TF_OD_API_MODEL_FILE = "210325_3rd_outputfloat_640640_quantized.tflite";
                break;
            case 3:
                TF_OD_API_IS_QUANTIZED = false;
                TF_OD_API_MODEL_FILE = "201223_640640_not_quantized_metaadded.tflite";
                break;
            case 4:
                TF_OD_API_IS_QUANTIZED = false;
                TF_OD_API_MODEL_FILE = "210324_640640_not_quantized.tflite";
                break;
            case 5:
                TF_OD_API_IS_QUANTIZED = false;
                TF_OD_API_MODEL_FILE = "210325_3rd_640640_not_quantized.tflite";
                break;
        }

        // If it is not quantized, raise rate of detection.
        if (!TF_OD_API_IS_QUANTIZED)
            MINIMUM_CONFIDENCE_TF_OD_API = 0.80f;

        // Initiate machine learning model
        try {
            detector =
                    TFLiteObjectDetectionAPIModel.create(
                            this.context,
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED,
                            TF_OD_CPU_OR_GPU);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public ArrayList<Output_Data.recognitions> process(Bitmap given_bitmap) {
        this.output_data.outputs.clear();
        Bitmap calc_bitmap = given_bitmap.copy(Bitmap.Config.ARGB_8888, true);

        // Create an ImageProcessor with all ops required.
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .build();

        // Create a TensorImage object. This creates the tensor of the corresponding
        // tensor type (uint8 in this case) that the TensorFlow Lite interpreter needs.
        TensorImage tImage;
        if (TF_OD_API_IS_QUANTIZED)
            tImage = new TensorImage(DataType.UINT8);
        else
            tImage = new TensorImage(DataType.FLOAT32);

        // Analysis code for every frame
        // Preprocess the image
        tImage.load(calc_bitmap);
        tImage = imageProcessor.process(tImage);

        try {
            final List<Detector.Recognition> results = detector.recognizeImage(tImage.getBitmap());

            // add logs for time in loading model
//            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
//            Log.v(TAG, "inferencing tacked time : " + lastProcessingTimeMs);
//            inferencing_time_list.add(lastProcessingTimeMs);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;

            int obj_cnt = 0;
            for (Detector.Recognition result : results) {
                final RectF location = result.getLocation();
                if (result.getConfidence() > minimumConfidence) {
                    obj_cnt++;
                    location.set(relocate_coords(location, calc_bitmap.getWidth(), calc_bitmap.getHeight()));
                    this.output_data.add(result.getConfidence(), location);
                }
            }
            Log.v(TAG, "size : " + results.size());
            if (obj_cnt == 0) {
                // if size == 0, add 0 values in output data
                this.output_data.add(0, new RectF(-1, -1, -1, -1));
            }
            Log.v(TAG, "valid objects : " + obj_cnt);

        } catch (Exception e) {
            e.printStackTrace();
        }

        return this.output_data.outputs;
    }

    /**
     * In the detector, every size of found coordinates are based on detector input size.
     * so we reconstruct its coordinates into original image size.
     */
    protected RectF relocate_coords(RectF from, int width, int height) {
        RectF to = new RectF();

        to.set(from.left * width / TF_OD_API_INPUT_SIZE, from.top * height / TF_OD_API_INPUT_SIZE,
                from.right * width / TF_OD_API_INPUT_SIZE, from.bottom * height / TF_OD_API_INPUT_SIZE);

        return to;
    }

}
