package com.ml_gram.project_coordination_final;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.media.MediaMetadataRetriever;
import android.os.AsyncTask;
import android.os.SystemClock;
import android.util.Log;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.TextView;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.max;

public class DetectAndWrite_2 extends AsyncTask<Integer, Bitmap, Boolean> {
    private String TAG = "DetectAndWrite";
    private ImageView imageView;
    private ProgressBar progressBar;
    private Context context;
    private MediaMetadataRetriever m;
    private TextView textView;
    private SeekBar seekBar;

    private List<Bitmap> bitmapArrayList;
    private Bitmap ResultBitmap;
    private ArrayList<RectF> coords_list;
    private ArrayList<Long> inferencing_time_list;
    private ArrayList<Float> detection_rate_list;
    private ArrayList<Integer> state_list;
    private int total_frames;

    private boolean TF_OD_CPU_OR_GPU = true;
    private static final int TF_OD_API_INPUT_SIZE = 640;
    private static final String TF_OD_API_LABELS_FILE = "coordination_labels.txt";
    private boolean TF_OD_API_IS_QUANTIZED = true;
    private String TF_OD_API_MODEL_FILE = "201223_640640_quantized_metaadded.tflite";


    // Minimum detection confidence to track a detection.
    private static float MINIMUM_CONFIDENCE_TF_OD_API = 0.70f;
    //    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.80f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final float TEXT_SIZE_DIP = 10;
    private Detector detector;

    //// for object detector lib api [END]
    public DetectAndWrite_2(Context context, ImageView views, ProgressBar progressBar, MediaMetadataRetriever mediaMetadataRetriever, List<Bitmap> bitmapArrayList, ArrayList<RectF> coords_list, TextView textView, ArrayList<Integer> state_list, SeekBar seekBar, int model_index) {
        this.context = context;
        this.imageView = views;
        this.progressBar = progressBar;
        this.m = mediaMetadataRetriever;
        this.bitmapArrayList = bitmapArrayList;
        this.coords_list = coords_list;
        this.textView = textView;
        this.state_list = state_list;
        this.seekBar = seekBar;

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
            MINIMUM_CONFIDENCE_TF_OD_API = 0.70f;

        imageView.setScaleType(ImageView.ScaleType.FIT_XY);
        inferencing_time_list = new ArrayList<>();
        detection_rate_list = new ArrayList<>();
    }

    @Override
    protected void onPreExecute() {
        Log.v(TAG, "on PreExecute");
        Log.v(TAG, "Selected mode : " + TF_OD_API_MODEL_FILE);
        super.onPreExecute();
    }

    public int dtw_hough(Bitmap bitmap_image, Rect loc, int offset) {
        int ret = 0;
        long startTime;
        long lastProcessingTimeMs;

        startTime = SystemClock.uptimeMillis();

        Rect new_loc = new Rect(loc.x - offset, loc.y - offset, loc.width + offset*2, loc.height + offset*2);

        // 시도하려는 것은, bitmap 객체를 Mat 변환 후
        // Mat의 일부 (loc 의 좌표를 기준으로 자른 일부 이미지) - Gray 변경 후 Hough 통과 시켜 Location 확보하기
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap_image, mat);

        // 여기는 Parsing Test 의 일부. 여기서부터 Test Code 임.
//        Rect rectCrop = new Rect(50, 50, 100, 100);
        Mat mat_pased = mat.submat(new_loc);

        Imgproc.cvtColor(mat_pased, mat_pased, Imgproc.COLOR_RGBA2GRAY);
        Mat output = new Mat();

        // Need to check if Gaussian blur has to be done before hough circles or not. (on server, enabled gaussian blur before do Hough)
        // 720 1600 only
        Imgproc.HoughCircles(mat_pased, output, Imgproc.HOUGH_GRADIENT, 1, 10, 150, 30, 18, 22);
        // With V30 model
//        Imgproc.HoughCircles(mat_pased, output, Imgproc.HOUGH_GRADIENT, 1, 10, 150, 20, 16, 22);

        // Turn mat_pased mat back to color
        Imgproc.cvtColor(mat_pased, mat_pased, Imgproc.COLOR_GRAY2BGRA);

        Log.v(TAG, "output num " + output.cols());
        for (int i = 0; i < output.cols(); i++) {
            double[] circle = output.get(0, i);
            double x = circle[0];
            double y = circle[1];
            double radius = circle[2];
            Log.v(TAG, "[x,y,r] =  [" +  x + "," + y + "," + radius);
            Imgproc.circle(mat_pased, new Point(x, y), (int) (radius * 1.1), new Scalar(255, 0, 0), 2);
        }

        Bitmap output_bitmap = Bitmap.createBitmap(mat_pased.cols(), mat_pased.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat_pased, output_bitmap);

        publishProgress(output_bitmap);
        bitmapArrayList.add(output_bitmap);

        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
        Log.v(TAG, "Hough calculation time : " + lastProcessingTimeMs);

        return ret;
    }
    @Override
    protected Boolean doInBackground(Integer... integers) {
        long startTime;
        long lastProcessingTimeMs;
        int state_idx = 0, state_cnt = 0, state_swt_cnt = 0;

        Log.v(TAG, "do In Background ");

        for (Integer integer : integers) {
            Log.v(TAG, "received : " + integer.toString());
            TF_OD_CPU_OR_GPU = (integer != 0);
        }

        // Initiate machine learning model
        try {
            startTime = SystemClock.uptimeMillis();
            detector =
                    TFLiteObjectDetectionAPIModel.create(
                            this.context,
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED,
                            TF_OD_CPU_OR_GPU);
            // add logs for time in loading model
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            Log.v(TAG, "loading model time : " + lastProcessingTimeMs);
        } catch (Exception e) {
            e.printStackTrace();
        }

        total_frames = Integer.parseInt(m.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT));

//        Bitmap bitmap_t = m.getFrameAtTime(500 * 1000, MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
//
//        Mat gray = new Mat();
//        Utils.bitmapToMat(bitmap_t, gray);
//
//        Log.v(TAG, "rows : " + gray.rows() + " cols : " + gray.cols());
//        dtw_hough(bitmap_t, new Rect(500, 1500, 100, 100), 0);


        if (true) {
//        for (int i = 0; i < duration ; i+=100) {
//            Log.v(TAG, "iteration start: " + i);
//            Bitmap bitmap = m.getFrameAtTime(i * 1000, MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
            for (int i = 0; i < total_frames; i++) {

                // scienario for skip frames for state switch.
                if (state_idx == 0) {
                    state_cnt = (state_cnt + 1) % 5;
                    if (state_cnt > 0) {
//                    Log.d(TAG, i + " state_idx [ " + state_idx + " ] " + "state_cnt [ " + state_cnt + " ] || SKIP");
                        continue;
                    } else {
//                    Log.d(TAG, i + "state_idx [ " + state_idx + " ] " + "state_cnt [ " + state_cnt + " ] || PROCESS");
                    }
                } else if (state_idx == 2) {
                    state_cnt = (state_cnt + 1) % 2;
                    if (state_cnt > 0) {
//                    Log.d(TAG, i + "state_idx [ " + state_idx + " ] " + "state_cnt [ " + state_cnt + " ] || SKIP");
                        continue;
                    } else {
//                    Log.d(TAG, i + "state_idx [ " + state_idx + " ] " + "state_cnt [ " + state_cnt + " ] || PROCESS");
                    }
                } else if (state_idx == 1) {
//                Log.d(TAG, i + "state_idx [ " + state_idx + " ] " + "state_cnt [ " + state_cnt + " ] || PROCESS");
                }
                startTime = SystemClock.uptimeMillis();
                Bitmap bitmap = m.getFrameAtIndex(i);
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                Log.v(TAG, "preprocessing(total image) time : " + lastProcessingTimeMs);

                startTime = SystemClock.uptimeMillis();
                bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
//            Log.v(TAG, "width : " + bitmap.getWidth());
//            Log.v(TAG, "height : " + bitmap.getHeight());

                // Initialization code
                // Create an ImageProcessor with all ops required. For more ops, please
                // refer to the ImageProcessor Architecture section in this README.
                ImageProcessor imageProcessor = new ImageProcessor.Builder()
                        .add(new ResizeOp(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                        .build();
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                Log.v(TAG, "preprocessing(image processor) time : " + lastProcessingTimeMs);

                startTime = SystemClock.uptimeMillis();
                // Create a TensorImage object. This creates the tensor of the corresponding
                // tensor type (uint8 in this case) that the TensorFlow Lite interpreter needs.
                TensorImage tImage;
                if (TF_OD_API_IS_QUANTIZED)
                    tImage = new TensorImage(DataType.UINT8);
                else
                    tImage = new TensorImage(DataType.FLOAT32);

                // Analysis code for every frame
                // Preprocess the image
                tImage.load(bitmap);
                tImage = imageProcessor.process(tImage);

                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                Log.v(TAG, "preprocessing(timage) time : " + lastProcessingTimeMs);
                try {
                    startTime = SystemClock.uptimeMillis();

                    final List<Detector.Recognition> results = detector.recognizeImage(tImage.getBitmap());
//                final List<Detector.Recognition> results = detector.recognizeImage(tImage);

                    // add logs for time in loading model
                    lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                    Log.v(TAG, "inferencing tacked time : " + lastProcessingTimeMs);
                    inferencing_time_list.add(lastProcessingTimeMs);

                    ResultBitmap = Bitmap.createBitmap(tImage.getBitmap());
                    final Canvas canvas = new Canvas(ResultBitmap);
                    final Paint paint = new Paint();
                    paint.setColor(Color.RED);
                    paint.setStyle(Paint.Style.STROKE);
                    paint.setStrokeWidth(2.0f);
                    paint.setTextSize(20);
                    float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;

//                for (final Detector.Recognition result : results) {
//                    final RectF location = result.getLocation();
//                    if (location != null && result.getConfidence() >= minimumConfidence) {
//                        canvas.drawRect(location, paint);
//                        canvas.drawText(String.format("%.2f", result.getConfidence()), location.right, location.bottom, paint);
//
//                        result.setLocation(location);
//                        coords_list.add(location);
//                        Log.v(TAG, "results = " + results.get(0));
//                    }
//                }

                    int obj_cnt = 0;
                    for (Detector.Recognition result : results) {
                        final RectF location = result.getLocation();
                        if (result.getConfidence() > minimumConfidence) {
                            // Test Code for debugging [START]
                            canvas.drawRect(location, paint);
                            canvas.drawText(String.format("%.2f", result.getConfidence()), location.right, location.bottom, paint);
                            // Test Code for debugging [END]
                            obj_cnt++;
                        }
                    }
                    Log.v(TAG, "valid objects : " + obj_cnt);

                    final Detector.Recognition result = results.get(0);
                    final RectF location = result.getLocation();
                    if (false) {
                        if (obj_cnt > 0) {
                            Mat gray = new Mat();
                            Utils.bitmapToMat(bitmap, gray);

                            Log.v(TAG, "rows : " + gray.rows() + " cols : " + gray.cols());
                            Log.v(TAG, "location.left : " + location.left + " location.top : " + location.top);
                            Log.v(TAG, "location.right : " + location.right + " location.bottom : " + location.bottom);
                            Log.v(TAG, "location.width : " + location.width() + " location.height : " + location.height());
                            Log.v(TAG, "x : " + (int) ((location.left/640) * gray.cols()) + " y :  " + (int) ((location.top/640) * gray.rows()));
                            try {
                                dtw_hough(bitmap, new Rect((int) ((location.left/640) * gray.cols()), (int) ((location.top/640) * gray.rows()),
                                        (int) (location.width()/640 * gray.cols()), (int) (location.height()/640 * gray.rows())), 20);
                                Thread.sleep(1000);
                            } catch (android.content.ActivityNotFoundException e) {
                                e.printStackTrace();
                                break;
                            }
//                            break;
                        }
                    }
                    if (location != null) {
                        if (obj_cnt == 1) {
                            detection_rate_list.add(result.getConfidence());
                            ;
//                        canvas.drawRect(location, paint);
//                        canvas.drawText(String.format("%.2f", result.getConfidence()), location.right, location.bottom, paint);
                        } else {
                            if (obj_cnt == 0)
                                location.set(new RectF(-1, -1, -1, -1));
                            // If there are many points which detected by detector, we choose only 1 best point of it and adapt it to script.
//                        else
//                            location.set(new RectF(-2, -2, -2, -2));
                        }
                        result.setLocation(location);
                        coords_list.add(location);
                        Log.v(TAG, "results = " + results.get(0) + " coords = " + coords_list.get(coords_list.size() - 1));

                        // If pressed event occur, state switch from 0 to 2
                        if (location.centerX() > 0) {
                            if (state_idx == 0) {
                                state_idx = 1;
                                state_cnt = 0;
                            } else if (state_idx == 1) {
                                state_swt_cnt++;
                                if (state_swt_cnt == 3) {
                                    state_idx = 2;
                                    state_swt_cnt = 0;
                                    state_cnt = 0;
                                }
                            }
                        } else if (location.centerX() == -1) { // if released.
                            if (state_idx != 0) {
                                state_idx = max(0, (state_idx - 1)) % 3;
                                state_cnt = 0;
                            }
                        }

                        state_list.add(state_idx);
                    }

                } catch (Exception e) {
                    e.printStackTrace();
                }
                bitmapArrayList.add(ResultBitmap);
                publishProgress(ResultBitmap);
                progressBar.setProgress(i);

//            Log.v(TAG, "iteration end: " + i);
            }
        }

        seekBar.setMax(bitmapArrayList.size());

        return null;
    }

    @Override
    protected void onProgressUpdate(Bitmap... bitmaps) {
//        Log.v(TAG, "on Progress Update, length : " + bitmaps.length);

        imageView.setImageBitmap(bitmaps[0]);

        long average = 0;
        float detect_rate = 0;
        for (long time : inferencing_time_list)
            average += time;

        for (Float detection_rate : detection_rate_list)
            detect_rate += detection_rate;

        textView.setText("Average inferencing time / Frame : " + average / inferencing_time_list.size() + " ms\n");
        textView.append("Average Detection rate / target : " + detect_rate * 100 / detection_rate_list.size() + " %");
        super.onProgressUpdate(bitmaps);
    }

    @Override
    protected void onPostExecute(Boolean aBoolean) {
        Log.v(TAG, "on Post Execute");

        imageView.setImageBitmap(bitmapArrayList.get(bitmapArrayList.size() - 1));

        progressBar.setProgress(total_frames);
        m.release();

        super.onPostExecute(aBoolean);
    }
}
