package com.ml_gram.project_coordination_final;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.InputStream;

public class OpencvActivity extends AppCompatActivity {
    final String TAG = "OpenCV_Activity";
    ImageView imageView_open;
//    ImageView imageView_open2;
    Button btn_open_image;
    Button btn_toggle;
    private boolean isOpenCVLoaded = false;
    Mat gray;
    Bitmap imageBitmap;
    int toggle = 0;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_opencv);

        imageView_open = findViewById(R.id.imageView_opencv);
//        imageView_open2 = findViewById(R.id.imageView_opencv2);
        btn_open_image = findViewById(R.id.button_opencv_files);
        btn_toggle = findViewById(R.id.button_opencv_toggle);


        btn_open_image.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.v(TAG, "Browse Clicked");

                Intent i = new Intent(Intent.ACTION_PICK);
                i.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
                try {
                    startActivityForResult(i, 0);
                } catch (android.content.ActivityNotFoundException e) {
                    e.printStackTrace();
                }
            }
        });

        btn_toggle.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.v(TAG, "toggle clicked");
                if (toggle++ %2 == 1) {
                    try {
                        Log.v(TAG, "gray information" + gray.cols() + gray.rows());
                        Bitmap grayBitmap = Bitmap.createBitmap(gray.cols(), gray.rows(), Bitmap.Config.ARGB_8888);
                        Utils.matToBitmap(gray, grayBitmap);
                        imageView_open.setImageBitmap(grayBitmap);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                } else {
                    imageView_open.setImageBitmap(imageBitmap);
                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK) {

            imageView_open.setScaleType(ImageView.ScaleType.FIT_XY);
//            imageView_open2.setScaleType(ImageView.ScaleType.FIT_XY);

            try {
                InputStream in = getContentResolver().openInputStream(data.getData());
                imageBitmap = BitmapFactory.decodeStream(in);
                in.close();

//                imageView_open.setImageBitmap(imageBitmap);

                if (!isOpenCVLoaded) {
                    Log.v(TAG, "open cv not loaded!!");
                    return;
                }

                // 0) count tacked time
                long startTime = SystemClock.uptimeMillis();

                // 1) image transform from color to gray scale
                Utils.bitmapToMat(imageBitmap, gray);
                Log.v(TAG, "rows : " + gray.rows() + "/ cols : " + gray.cols());
                if (gray.rows() != 1440 && gray.cols() != 720) {
                    Imgproc.resize(gray, gray, new Size(720, 1440), 0, 0, Imgproc.INTER_AREA);
                    Log.v(TAG, "Resized rows : " + gray.rows() + "cols : " + gray.cols());
                }

                Imgproc.cvtColor(gray, gray, Imgproc.COLOR_RGBA2GRAY);

                // 2) Image blur (get rid of noise)
                Imgproc.GaussianBlur(gray, gray, new Size(3, 3), 0);

                // 3) use Hough circles which includes canny edge detector to find out where circles are
                Mat output = new Mat();
//                3-1) values for V50
//                Imgproc.HoughCircles(gray, output, Imgproc.HOUGH_GRADIENT, 1, 200, 150, 20, 20, 22);

//                3-2) Values for V30
//                Imgproc.HoughCircles(gray, output, Imgproc.HOUGH_GRADIENT, 1, 100, 150, 40, 20, 22);

//                3-2) Values for Kiwoo Phone
                Imgproc.HoughCircles(gray, output, Imgproc.HOUGH_GRADIENT, 1, 100, 150, 30, 16, 20);

                long endTime = SystemClock.uptimeMillis() - startTime;
                Log.v(TAG, "output(opencv) tacked time : " + endTime);

                Log.v(TAG, "output num " + output.cols());
                for (int i = 0; i < output.cols(); i++) {
                    double[] circle = output.get(0, i);
                    double x = circle[0];
                    double y = circle[1];
                    double radius = circle[2];
                    Log.v(TAG, "[x,y,r] =  [" +  x + "," + y + "," + radius);
                    Imgproc.circle(gray, new Point(x, y), (int) (radius * 1.1), new Scalar(127, 127, 127), 5);

                }

                Bitmap gray_bitmap = Bitmap.createBitmap(gray.cols(), gray.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(gray, gray_bitmap);

                imageView_open.setImageBitmap(gray_bitmap);

            } catch (Exception e) {
                e.printStackTrace();
            }

        }
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.v(TAG, "OpenCV Load success");
                    if (gray == null)
                        gray = new Mat();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.v(TAG, "Internal OpenCV Library not found, using Opencv manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.v(TAG, "OpenCV library found insde package, using it!!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            isOpenCVLoaded = true;
        }
    }
}
