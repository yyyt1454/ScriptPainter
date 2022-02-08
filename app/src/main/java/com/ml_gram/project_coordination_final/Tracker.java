package com.ml_gram.project_coordination_final;

import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.lang.reflect.Array;
import java.util.ArrayList;

/**
 * Light-weight Hough Tracker
 * [jobs]
 * - Find circle of specific size (which points at cursor) in the given bounding box.
 * - bounding box should be adaptive due to its previous speed.
 *
 * [inputs / outputs]
 * - input : Bitmap frame, Rect coords (for previous cursor location), frm_cnt (for estimating consecutiveness of frames)
 * - output : objectness, Rect coords, score
 *
 */

public class Tracker {
    private String TAG = "Tracker";

    /**
     * Output data
     */
    private Output_Data output_data;
    public class Output_Data {
        public class recognitions {
            public float confidence;
            public RectF location;
            public recognitions (float confidence, RectF location) {
                this.confidence = confidence;
                this.location = location;
            }
        }
        public ArrayList<recognitions> outputs;

        public Output_Data () {
            this.outputs = new ArrayList<>();
        }

        public void add (float confidence, RectF location) {
            recognitions recogs = new recognitions(confidence, location);
            this.outputs.add(recogs);
        }
    }

    /**
     * Definitions.
     */
    private static final int bb_offset = 20;
    private int bb_adaptive_offset = bb_offset;

    /**
     * Variables
     */
    private RectF bb_location;
    Mat given_mat;
    Rect bb_rect;
    Mat bb_mat_color;
    Mat bb_mat_gray;
    Mat output_mat;

    float confidence;
    // for recording coordinates of all in tracker
    public ArrayList<double[]> saved_list;

    /**
     * Constructor
     */
    public Tracker() {
        this.output_data = new Output_Data();
        this.bb_location = new RectF();
        this.saved_list = new ArrayList<>();
    }

    /**
     * Methods.
     * 1) setBB : set bounding box of current frame. (its value should be same with lastly found cursor location)
     * 2) process : process hough circles to find cursor in given image with adaptive bounding box.
     * 3) get bounding box : to show size of temporal bounding box.
     */
    public void set_bounding_box (RectF bb_location) {
        this.bb_location.set(bb_location);
    }

    public ArrayList<Output_Data.recognitions> process (Bitmap given_bitmap, int frm_idx) {
        this.output_data.outputs.clear();
        double [] circle = new double[]{0,0,0};
        float [] center = {bb_location.centerX(), bb_location.centerY()};
        double x = 0, y = 0, radius = 0;
        this.given_mat = new Mat();
        this.bb_rect = new Rect();
        this.bb_mat_color = new Mat();
        this.bb_mat_gray = new Mat();
        this.output_mat = new Mat();
        double dx = 0, dy = 0;
        double [] last_saved, last_prev_saved;
        double y_estimate = 0, x_estimate = 0;

        Bitmap calc_bitmap = given_bitmap.copy(Bitmap.Config.ARGB_8888, true);

        /**
         * 1) Transform given bitmap to mat, which is the type that used in hough transform.
         */
        Utils.bitmapToMat(calc_bitmap, given_mat);
        /**
         * 2-1) set appropriate bounding box which can be adaptable due to its previous speed,
         * Start from HERE!!!!
         * 1) Adaptable bounding box
         * 2) if offset if out of range (bigger than full size of img, return to work with detector
         */
        // 1) adaptable bounding box
        // 2) if offset is out of range, return.
        if (this.saved_list.size() < 2) {
            Log.v(TAG, "saved list is not valid save length: " + this.saved_list.size());
            bb_adaptive_offset = bb_offset;
            Log.v(TAG, "using default offset : " + bb_adaptive_offset);
        } else {
            last_saved = this.saved_list.get(this.saved_list.size() - 1);
            last_prev_saved = this.saved_list.get(this.saved_list.size() - 2);

            if ((last_saved[0] - last_prev_saved[0] < 5) && (frm_idx - last_saved[0] < 5)) {
                y_estimate = Math.abs(last_saved[1] - last_prev_saved[1]) + Math.abs(last_saved[1]);
                x_estimate = Math.abs(last_saved[2] - last_prev_saved[2]) + Math.abs(last_saved[2]);

                if ((y_estimate > bb_adaptive_offset * 0.75) || (x_estimate > bb_adaptive_offset * 0.75)) {
                    if (bb_adaptive_offset >= 80) {
                        Log.v(TAG, "Expanded enough, not expanding anymore");
                    } else {
                        Log.v(TAG, "Expanding from " + bb_adaptive_offset + " to " + bb_adaptive_offset * 2);
                        bb_adaptive_offset *= 2;
                    }
                } else if ((bb_adaptive_offset > bb_offset) && (y_estimate < bb_adaptive_offset * 0.25) && (x_estimate < bb_adaptive_offset * 0.25)) {
                    if (bb_adaptive_offset > bb_offset) {
                        Log.v(TAG, "Shrink from " + bb_adaptive_offset + " to " + bb_adaptive_offset / 2);
                        bb_adaptive_offset /= 2;
                    } else {
                        Log.v(TAG, "Shrink from " + bb_adaptive_offset + " to " + bb_adaptive_offset / 2);
                        bb_adaptive_offset = bb_offset;
                    }
                }

            } else {
                Log.v(TAG, "saved list" + last_prev_saved[0] + " / " + last_saved[0] + "are not consecutive, using default offset");
                bb_adaptive_offset = bb_offset;
            }
        }

        if ((bb_location.left - bb_adaptive_offset < 0) ||
                (bb_location.top - bb_adaptive_offset < 0) ||
                ((int)(bb_location.right + bb_adaptive_offset) > (calc_bitmap.getWidth() - 1)) ||
                ((int)(bb_location.bottom + bb_adaptive_offset) > (calc_bitmap.getHeight() - 1))
        ){
            Log.v(TAG, "out of range, execute with detector");
            Log.v(TAG, "bb left / right / top / bottom / bb_offset : " + bb_location.left + " / " + bb_location.right + " / " + bb_location.top + " / "
             + bb_location.bottom + " / " + bb_offset);
            bb_adaptive_offset = bb_offset;
            this.output_data.add(0, new RectF((float)(bb_location.left + dx),
                    (float)(bb_location.top + dy),
                    (float)(bb_location.right + dx),
                    (float)(bb_location.bottom + dy)
            ));
            return this.output_data.outputs;
        }


        /**
         * 2-2) set bounding box rect to parse from given image mat.
         * param is (left, top, width, height)
         */
//         if bounding box is out of range, it has to be shrink as default value, even with out of range occurs, return to use detector.

        double [] bb = {(bb_location.left - bb_adaptive_offset) >= 0 ? (bb_location.left - bb_adaptive_offset) : 0,
                (bb_location.top - bb_adaptive_offset) >= 0 ? (bb_location.top - bb_adaptive_offset) : 0,
                bb_location.width() + bb_adaptive_offset * 2,
                bb_location.height() + bb_adaptive_offset * 2
        };
        bb_rect.set(bb);
        bb_mat_color = given_mat.submat(bb_rect);
        Log.v(TAG, "bb_locations : [" + bb_location.left + "," + bb_location.top + "," + bb_location.right + "," + bb_location.bottom + "]");
        Log.v(TAG, "bb width / height : " + bb_mat_color.width() + " / " + bb_mat_color.height());

        /**
         * 3) calculate HoughCircles within bounding box area
         */
        Imgproc.cvtColor(bb_mat_color, bb_mat_gray, Imgproc.COLOR_RGBA2GRAY);
        if (given_bitmap.getWidth() <= 800)
            Imgproc.HoughCircles(bb_mat_gray, output_mat, Imgproc.HOUGH_GRADIENT, 1, 10, 150, 30, 18, 22);
        else
            Imgproc.HoughCircles(bb_mat_gray, output_mat, Imgproc.HOUGH_GRADIENT, 1, 10, 150, 30, 26, 30);

        Log.v(TAG, "hough outputs : " + output_mat.cols());
        // no circle or redundant circles are found in BB. it needs detector to find the cursor accurately.
        if (output_mat.cols() == 1) {
            circle = output_mat.get(0, 0);
            x = circle[0];
            y = circle[1];
            radius = circle[2];
            confidence = 100;
            Log.v(TAG, "[x,y,r] =  [" +  x + "," + y + "," + radius);
        } else {
            confidence = 0;
        }

        // found coordinates should be calculated with previous center point value.
        dy = y - (bb_rect.height / 2);
        dx = x - (bb_rect.width / 2);

        this.output_data.add(confidence, new RectF((float)(bb_location.left + dx),
                (float)(bb_location.top + dy),
                (float)(bb_location.right + dx),
                (float)(bb_location.bottom + dy)
        ));

        if (confidence == 100) {
            double [] tmp_list = {frm_idx, dy, dx};
            this.saved_list.add(tmp_list);
        }

        return this.output_data.outputs;
    }

    public int get_bb_offset () {
        return this.bb_adaptive_offset;
    }

}
