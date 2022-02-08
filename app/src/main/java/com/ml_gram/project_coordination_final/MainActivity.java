package com.ml_gram.project_coordination_final;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Point;
import android.graphics.RectF;
import android.media.MediaMetadataRetriever;
import android.media.MediaPlayer;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Display;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

import wseemann.media.FFmpegMediaMetadataRetriever;
public class MainActivity extends AppCompatActivity {
    private final String TAG = "Main";
    /**
     * View group for main activity
     * */
    public RadioGroup radioGroup;
    public RadioButton btn_radio_cpu, btn_radio_gpu;
    public Button button_browse, button_toggle, button_graph, button_script;
    public ImageView imageView;
    public ProgressBar progressBar;
    public SeekBar seekBar;
    public TextView textView;
    public Spinner spinner;

    // for old version
//    public DetectAndWrite_2 detectAndWrite;
    // for new version
    public ScriptPainter scriptPainter;
    public FrameManager frameManager;
    public Dnn_Detector dnn_detector;
    public Tracker tracker;

    /**
    * Variables for saving processed data (old)
    * */
    private List<Bitmap> bitmapArrayList;
    private ArrayList<RectF> coords_results;
    private ArrayList<RectF> coords_modified_results;
    private ArrayList<Integer> state_results;
    private ArrayList<String> select_quantized;
    private ArrayList<Long> tacked_time_total;

    /**
     *  Data_list for saving frame_num, state, coords (new)
      */
    private class Data {
        int frame_num;
        int state;
        RectF coords;
        public Data (Integer frame_num, Integer state, RectF coords) {
            this.frame_num = frame_num;
            this.state = state;
            this.coords = coords;
        }
    }
    private ArrayList<Data> data_list = new ArrayList<>();

    /**
     * indexes for describe data type
     * */
    private static final int SELECT_MOVIE = 0;
    private int model_index = 0;
    private static final int STATE_NO_OBJECT = 0;
    private static final int STATE_OBJECT_DETECTED = 1;
    private static final int STATE_DETECTED_STABILIZE = 2;
    private boolean isOpenCVLoaded = false;

    public class ScriptPainter extends AsyncTask<MediaMetadataRetriever, Bitmap, Boolean> {
        /**
         * - Need input : Video file
         * - Output : data_list (accumulated list which composed of frame num, state, coords)
         * 1) Frame manager
         * 2) DNN detector
         * 3) Lightweight tracker
         * 4) Script generator
         * -> script generator isn't mandatory, since it can be generated out of async task.
         *
         */

        /**
         * Variables for processing data.
         * status = detection status + tracking status will be shown for next frame
         */
        int status = 0;
        int frm_idx = 0;
        int total_frames = 0;
        int frm_width, frm_height = 0;

        @Override
        protected void onProgressUpdate(Bitmap... values) {
            imageView.setImageBitmap(values[0]);

            long average = 0;
            for (long val : tacked_time_total) {
                average += val;
            }

//            "Average Inferencing Time : \nAverage Detection rate / target :"
            textView.setText("Average Inferencing Time : " + average * 2 / (3 * tacked_time_total.size()) + " ms\n");
            super.onProgressUpdate(values);
        }

        /**
         * get proper sampled frame through frame manager, and compute via detector or tracker.
         * @param mediaMetadataRetrievers
         * @return
         */
        @Override
        protected Boolean doInBackground(MediaMetadataRetriever... mediaMetadataRetrievers) {
            // Thread priority to highest
            Thread.currentThread().setPriority(Thread.MAX_PRIORITY);

            // for frame manager
            MediaMetadataRetriever m = mediaMetadataRetrievers[0];
            frameManager = new FrameManager(m);
            boolean need_rewind = false;
            int rewind = 0;
            total_frames = Integer.parseInt(m.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT));
            frm_width = Integer.parseInt(m.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH));
            frm_height = Integer.parseInt(m.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT));
//            Log.d(TAG, "Total Frames : " + total_frames);

            // for initialize
            frm_idx = 0;
            data_list.clear();

            // for detector
            dnn_detector = new Dnn_Detector(getApplicationContext(), model_index, btn_radio_cpu.isChecked() ? false : true);
            ArrayList<Dnn_Detector.Output_Data.recognitions> dnn_output = new ArrayList<>();

            // for tracker
            tracker = new Tracker();
            ArrayList<Tracker.Output_Data.recognitions> tracker_output = new ArrayList<>();
            FrameManager.Output_Data sampled_data;

            // for painting
            final Paint paint = new Paint();
            paint.setColor(Color.GREEN);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(2.0f);
            paint.setTextSize(20);

            final Paint paint_tr = new Paint();
            paint_tr.setColor(Color.RED);
            paint_tr.setStyle(Paint.Style.STROKE);
            paint_tr.setStrokeWidth(2.0f);
            paint_tr.setTextSize(20);
            long startTime;
            long tacked_time;

            // for generating right coordinates
            Display display = getWindowManager().getDefaultDisplay();
            Point size = new Point();
            display.getRealSize(size);
//            Log.v(TAG, "size x : " + size.x + " size y : " + size.y);

            startTime = SystemClock.uptimeMillis();
            frameManager.preprocess();

            startTime = SystemClock.uptimeMillis() - startTime;
            Log.v(TAG, "Pre process : " + startTime);
            while (frm_idx < total_frames) {
                /**
                 * 1) Frame sampling process
                 * Through frame manager, samples proper frame in video file.
                 * There is state manager that manages 3 states (Initial, Detected, Stabilize) within frame manager.
                 * So detection, tracking result should be notified to frame manager to sample properly.
                 */
                if (frm_idx >= total_frames - frameManager.stateManager.Initial_sampling) {
                    break;
                }

                startTime = SystemClock.uptimeMillis();
                sampled_data = frameManager.process(status, need_rewind, rewind);
                frm_idx = sampled_data.frame_idx;
                Log.d(TAG, "frm_idx : " + frm_idx);
                tacked_time = SystemClock.uptimeMillis() - startTime;
                tacked_time_total.add(tacked_time);
                Log.v(TAG, "TIME [sampling] : " + tacked_time);

                /**
                 * 2) Frame allocation process
                 * Frame allocation to detector or tracker is decided via result status of detection, tracking status.
                 * if last frame was detected well, it could be consecutively tracked through tracker, so allocate frame to tracker.
                 * otherwise, allocate frame to detector.
                 */
                if (status == STATE_NO_OBJECT) {
                    startTime = SystemClock.uptimeMillis();
                    dnn_output = dnn_detector.process(sampled_data.result_bitmap);
                    tacked_time = SystemClock.uptimeMillis() - startTime;
                    tacked_time_total.add(tacked_time);
                    Log.v(TAG, "TIME [detector] : " + tacked_time);
                    /**
                     * 3) Rewind process.
                     * Rewind process is executed when object is shown in 'Initial state', rewind back frames
                     * and check consecutively if there is any missing point to enhance accuracy.
                     */
//                    Log.v(TAG, "size : " + dnn_output.size());
                    if (dnn_output.get(0).location.centerX() > 0) {
                        for (int i = frameManager.stateManager.Initial_sampling - 1; i >= 0; i--) {
                            sampled_data = frameManager.process(status, true, i);

                            dnn_output = dnn_detector.process(sampled_data.result_bitmap);
                            if (dnn_output.get(0).location.centerX() > 0) {
                                tracker.set_bounding_box(dnn_output.get(0).location);
                                status = STATE_OBJECT_DETECTED;
                                frameManager.stateManager.switch_state(true);
                                break;
                            }
                        }
                    }
                } else if (status == STATE_OBJECT_DETECTED) {
                    /**
                     * once it is detected, 3 consecutive frames are should be processed with detector
                     */
                    if (frameManager.stateManager.current_state.state == frameManager.stateManager.Detection_state) {
                        startTime = SystemClock.uptimeMillis();
                        dnn_output = dnn_detector.process(sampled_data.result_bitmap);
                        tacked_time = SystemClock.uptimeMillis() - startTime;
                        tacked_time_total.add(tacked_time);
                        Log.v(TAG, "TIME [detector] : " + tacked_time);
                        if (dnn_output.get(0).location.centerX() > 0) {
                            tracker.set_bounding_box(dnn_output.get(0).location);
                        } else {
                            status = STATE_NO_OBJECT;
                            frameManager.stateManager.switch_state(false);
                        }
                    }
                    /**
                     * Tracker will be checking cursors in "stabilize" state. if it doesn't find the cursor
                     * properly, detector should do its work.
                     */
                    else {
                        startTime = SystemClock.uptimeMillis();
                        tracker_output = tracker.process(sampled_data.result_bitmap, frm_idx);
                        tacked_time = SystemClock.uptimeMillis() - startTime;
                        tacked_time_total.add(tacked_time);
                        if (tracker_output.get(0).confidence > 0) {
                            tracker.set_bounding_box(tracker_output.get(0).location);
                        } else {
                            dnn_output = dnn_detector.process(sampled_data.result_bitmap);
                            if (dnn_output.get(0).location.centerX() > 0) {
                                tracker.set_bounding_box(dnn_output.get(0).location);
                            } else {
                                status = STATE_NO_OBJECT;
                                frameManager.stateManager.switch_state(false);
                            }
                        }
                    }
                }

                /**
                 * 4) coordinate accumulation, image showing process.
                 * should be re defined in OnProgress.
                 * */
                final Canvas canvas = new Canvas(sampled_data.result_bitmap);
                for (Dnn_Detector.Output_Data.recognitions outputs : dnn_output) {
                    if (outputs.confidence == 0) break;
                    canvas.drawRect(outputs.location, paint);
                    canvas.drawText(String.format("%.2f", outputs.confidence), outputs.location.right, outputs.location.bottom, paint);
                }
                for (Tracker.Output_Data.recognitions outputs : tracker_output) {
                    if (outputs.confidence == 0) break;
//                    canvas.drawRect(outputs.location, paint_tr);
                    // To show bouding box of cursor
                    RectF temp_bb = new RectF();
                    temp_bb.set(outputs.location.left - tracker.get_bb_offset(),
                            outputs.location.top - tracker.get_bb_offset(),
                            outputs.location.right + tracker.get_bb_offset(),
                            outputs.location.bottom + tracker.get_bb_offset());
                    canvas.drawRect(temp_bb, paint_tr);
                    canvas.drawCircle(outputs.location.centerX(), outputs.location.centerY(), (outputs.location.width()/2), paint_tr);
                    canvas.drawText(String.format("%.2f", outputs.confidence), outputs.location.right, outputs.location.bottom, paint_tr);
                }

                publishProgress(sampled_data.result_bitmap);

                /**
                 * Since detector, tracker returns same coordinate system which related with recorded image,
                 * it only needs to be converted as device related coordinate system to make as script.
                 */
                RectF tmp_output;
                if (tracker_output.size() > 0 && tracker_output.get(0).confidence == 100) { //tracker
                    Log.v(TAG, "[Final data owner] Tracker");
                    tmp_output = tracker_output.get(0).location;
                } else {
                    Log.v(TAG, "[Final data owner] Detector");
                    tmp_output = dnn_output.get(0).location;
                }
                RectF temp = relocate_coords(tmp_output, frm_width, frm_height, size.x, size.y);

                Data d = new Data(frm_idx, frameManager.stateManager.current_state.state, temp);

                data_list.add(d);
                bitmapArrayList.add(sampled_data.result_bitmap);
                progressBar.setProgress(frm_idx);

                dnn_output.clear();
                tracker_output.clear();
            }
            return false;
        }

        @Override
        protected void onPostExecute(Boolean aBoolean) {
            seekBar.setMax(bitmapArrayList.size() - 1);
            imageView.setImageBitmap(bitmapArrayList.get(bitmapArrayList.size() - 1));
            progressBar.setProgress(total_frames);
//            for (double [] a : tracker.test)
//                Log.v(TAG,"saved : " + a[0] + " / " + a[1] + " / " + a[2]);

            super.onPostExecute(aBoolean);
        }

        /**
         * relocate coordination due to its width & height.
         * given coordinates from dnn detector is compose of 640 / 640 inputs. so we manipulate its coordinates
         * to work properly.
         *
         * field of from : should be same size of given frame image.
         */
        protected RectF relocate_coords(RectF from, int from_width, int from_height, int to_width, int to_height) {
            RectF to = new RectF();

            if (from.left < 0) {
                to.set(from);
            } else {
                to.set(from.left * to_width / from_width, from.top * to_height / from_height,
                        from.right * to_width / from_width, from.bottom * to_height / from_height);
            }

            return to;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        init_UI();

        button_browse.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
//                Log.v(TAG, "Browse clicked");

                Intent i = new Intent(Intent.ACTION_PICK);
                i.setDataAndType(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, "video/*");
                try {
                    startActivityForResult(i, 0);
                } catch (android.content.ActivityNotFoundException e) {
                    e.printStackTrace();
                }
            }
        });

        button_toggle.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (bitmapArrayList.size() > 0) {
                    seekBar.setProgress(seekBar.getProgress() + 1);
                }
                else
                    Toast.makeText(MainActivity.this, "please inference before click toggle", Toast.LENGTH_SHORT).show();

            }
        });

        radioGroup.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                if (checkedId == R.id.btn_radio_cpu) {
                    Toast.makeText(MainActivity.this, "Inferencing with CPU", Toast.LENGTH_SHORT).show();
                } else if (checkedId == R.id.btn_radio_gpu) {
                    Toast.makeText(MainActivity.this, "Inferencing with GPU", Toast.LENGTH_SHORT).show();
                }
            }
        });

        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                model_index = (int) id;
                Log.v(TAG, id + " is selected, quantized : " + id);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                Log.v(TAG, "onprogress changed");
                imageView.setImageBitmap(bitmapArrayList.get(Math.min(seekBar.getProgress(), bitmapArrayList.size()-1)));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                Log.v(TAG, "onStartTrackingTouch");
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                Log.v(TAG, "onStopTrackingTouch");
            }
        });

        button_graph.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplicationContext(), GraphActivity.class);
                Log.v(TAG, "data_list size : " + data_list.size());

                coords_results.clear();
                coords_modified_results.clear();
                state_results.clear();
                for (int i = 0; i < data_list.size(); i++) {
                    coords_results.add(data_list.get(i).coords);
                    coords_modified_results.add(data_list.get(i).coords);
                    state_results.add(data_list.get(i).state);
                }

                int validation = 0;
                for (int i = 0; i < coords_modified_results.size(); i++) {
                    // when touch is pressed
                    if (coords_modified_results.get(i).centerX() > 0) {
                        validation++;
                    } else { // where touch is released
                        if (validation > 0 && validation < 5) {
                            // released before 5 consecutive frames are actually pressed!! so erase this trash points.
                            RectF zero = new RectF(-1, -1, -1, -1);
                            while (validation > -1) {
                                Log.v(TAG, "before : i : " + i + "[ " + coords_modified_results.get(i) + " ], vali : " + validation);
                                coords_modified_results.set(i - validation, zero);
                                Log.v(TAG, "After : i : " + i + "[ " + coords_modified_results.get(i) + " ], vali : " + validation);

                                // state result is used for modifying coordinations
                                state_results.set(i - validation, STATE_NO_OBJECT);
                                validation--;
                            }
                        }
                        validation = 0;
                    }
                }

                intent.putParcelableArrayListExtra("coords_results", coords_results);
                intent.putParcelableArrayListExtra("coords_modified_results", coords_modified_results);
                intent.putIntegerArrayListExtra("state_results", state_results);
                startActivity(intent);
            }
        });

        button_script.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
//                if (coords_results.size() == 0)
//                    Toast.makeText(MainActivity.this, "No coords found, plz inference first", Toast.LENGTH_SHORT).show();
//                else {
//
//                }
                ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, MODE_PRIVATE);

                String state = Environment.getExternalStorageState();
                if (state.equals(Environment.MEDIA_MOUNTED)) {
                    Log.v(TAG, "READ WRITE OK");
                    String Path = Environment.getExternalStorageDirectory().getAbsolutePath();
                    Log.v(TAG, "Path : " + Path);

                    File saveFile = new File(Path + "/my_script_dir");
                    Log.v(TAG, "exists = " + saveFile.exists());
                    if (!saveFile.exists())
                        saveFile.mkdir();

                    try {
                        BufferedWriter buf = new BufferedWriter(new FileWriter(saveFile + "/my_script.txt", false));

                        buf.append("type= user\n");
                        buf.append("speed= 1000\n");
                        buf.append("start data >>\n\n");

//                        buf.append("[frame no / state / coords]\n");
//                        for (int i = 0; i < data_list.size(); i++) {
//                            buf.append(data_list.get(i).frame_num + " / " + data_list.get(i).state + " / [" +
//                                    data_list.get(i).coords.centerX() + ", " + data_list.get(i).coords.centerY() + "\n");
//                        }
                        int sleep_cnt = 0;
                        for (int i = 0; i < data_list.size(); i++) {
                            Data temp_data = data_list.get(i);
                            if (temp_data.state == STATE_NO_OBJECT) {
                                sleep_cnt += 20 * frameManager.stateManager.Initial_sampling;
                            } else if (temp_data.state == STATE_DETECTED_STABILIZE) {
                                sleep_cnt += 20 * frameManager.stateManager.Stabilize_sampling;

                                if (data_list.get(i-1).state == STATE_DETECTED_STABILIZE) {
                                    double x_diff = Math.pow((temp_data.coords.centerX() - data_list.get(i-1).coords.centerX()), 2);
                                    double y_diff = Math.pow((temp_data.coords.centerY() - data_list.get(i-1).coords.centerY()), 2);

                                    double coords_diff = Math.sqrt(x_diff + y_diff);

                                    // It has to be tuned more
                                    if (coords_diff > 20) {
                                        buf.append("UserWait(" + sleep_cnt + ")\n");
                                        buf.append("DispatchPointer(0, 0, 2, " + temp_data.coords.centerX() + ", " + temp_data.coords.centerY() +
                                                ", 0, 0, 0, 0, 0, 0, 0)\n");
                                        sleep_cnt = 0;
                                    }
                                } else if (data_list.get(i-1).state == STATE_OBJECT_DETECTED) {
                                    buf.append("UserWait(" + sleep_cnt + ")\n");
                                    buf.append("DispatchPointer(0, 0, 0, " + temp_data.coords.centerX() + ", " + temp_data.coords.centerY() +
                                            ", 0, 0, 0, 0, 0, 0, 0)\n");
                                    sleep_cnt = 0;
                                }
                            } else if (temp_data.state == STATE_OBJECT_DETECTED) {
                                // If Press
                                if (data_list.get(i-1).state == STATE_NO_OBJECT) {
                                    buf.append("UserWait(" + sleep_cnt + ")\n");
                                    buf.append("DispatchPointer(0, 0, 0, " + temp_data.coords.centerX() + ", " + temp_data.coords.centerY() +
                                            ", 0, 0, 0, 0, 0, 0, 0)\n");
                                    sleep_cnt = 0;
                                } else if (data_list.get(i-1).state == STATE_DETECTED_STABILIZE) {
                                    // If Release
                                    buf.append("UserWait(" + sleep_cnt + ")\n");
                                    buf.append("DispatchPointer(0, 0, 1, " + data_list.get(i-1).coords.centerX() + ", " + data_list.get(i-1).coords.centerY() +
                                            ", 0, 0, 0, 0, 0, 0, 0)\n");
//                                    buf.append("UserWait(" + 50 + ")\n");
                                    sleep_cnt = 100;
                                }
                            }
                        }
                        buf.close();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                else if (state.equals(Environment.MEDIA_MOUNTED_READ_ONLY))
                    Log.v(TAG, "ONLY READ OK");
                else
                    Log.v(TAG, "cannot do somethinG");
            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && requestCode == SELECT_MOVIE) {
            MediaMetadataRetriever m = new MediaMetadataRetriever();
            m.setDataSource(this, data.getData());

            // check duration of media data
            MediaPlayer mediaPlayer = MediaPlayer.create(getBaseContext(), data.getData());
            int ms = mediaPlayer.getDuration();
            Log.v(TAG, "Video Duration (ms) : " + ms);
            Log.v(TAG, "Total Frames : " + m.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT));

            progressBar.setMax(Integer.parseInt(m.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)));
            bitmapArrayList.clear();
            coords_results.clear();
            state_results.clear();

            scriptPainter = new ScriptPainter();
            scriptPainter.execute(m);

            seekBar.setEnabled(true);
            button_toggle.setEnabled(true);
            button_graph.setEnabled(true);
            button_script.setEnabled(true);
        }
    }

    public void init_UI() {
        radioGroup = findViewById(R.id.radio_group);
        btn_radio_cpu = findViewById(R.id.btn_radio_cpu);
        btn_radio_gpu = findViewById(R.id.btn_radio_gpu);
        spinner = findViewById(R.id.spinner);
        button_browse = findViewById(R.id.button_browse);
        button_toggle = findViewById(R.id.button_toggle);
        button_graph = findViewById(R.id.button_graph);
        button_script = findViewById(R.id.button_script);
        imageView = findViewById(R.id.imageView);
        progressBar = findViewById(R.id.progressBar);
        textView = findViewById(R.id.textView);
        seekBar = findViewById(R.id.seekBar);
        bitmapArrayList = new ArrayList<Bitmap>();
        coords_results = new ArrayList<RectF>();
        coords_modified_results = new ArrayList<RectF>();
        state_results = new ArrayList<Integer>();
        tacked_time_total = new ArrayList<>();
        btn_radio_cpu.setChecked(true);
        button_toggle.setEnabled(false);
        button_script.setEnabled(false);
        button_graph.setEnabled(false);
        seekBar.setMax(0);
        seekBar.setEnabled(false);

        select_quantized = new ArrayList<>();
        select_quantized.add("Quantized_V30");
        select_quantized.add("Quantized_V30_Rainbow");
        select_quantized.add("Quantized_Rainbow");
        select_quantized.add("Not Quantized_V30");
        select_quantized.add("Not Quantized_V30_Rainbow");
        select_quantized.add("Not Quantized_Rainbow");
        ArrayAdapter arrayAdapter = new ArrayAdapter<>(getApplicationContext(), android.R.layout.simple_spinner_dropdown_item, select_quantized);
        spinner.setAdapter(arrayAdapter);

        imageView.setScaleType(ImageView.ScaleType.FIT_XY);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater menuInflater = getMenuInflater();
        menuInflater.inflate(R.menu.menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        switch (item.getItemId()) {
            case R.id.menu1_main:
                Toast.makeText(this, "Main Menu clicked!! do nothing", Toast.LENGTH_SHORT).show();
                break;
            case R.id.menu2_opencv:
                Intent intent = new Intent(MainActivity.this, OpencvActivity.class);
                startActivity(intent);
                break;
        }
        return super.onOptionsItemSelected(item);
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.v(TAG, "OpenCV Load success");
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