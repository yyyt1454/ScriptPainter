package com.ml_gram.project_coordination_final;

import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.util.Log;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

/**
 * Frame manager (sampling & rewinding)
 *  [jobs]
 *  - sampling
 *  - rewinding
 *  - omit proper sampled & rewinded frame / frame_num / proper state
 *
 *  [inputs / outputs]
 *  - input : Video file
 *  - output : Bitmap frame, Integer frame_idx, state state.
 */
public class FrameManager {
    String TAG = "Frame Manager";
    StateManager stateManager;

    // inputs
    MediaMetadataRetriever m;
    public List<Bitmap> all_bitmaps;
    // outputs
    public class Output_Data {
        Bitmap result_bitmap;
        Integer frame_idx;
        public Output_Data (Bitmap bitmap, Integer frame_idx) {
            this.result_bitmap = bitmap;
            this.frame_idx = frame_idx;
        }
    }

    // variables
    public static int frm_idx = 0;

    // variable for recursive preprocess
    public static int frm_saved = 0;
    public List<Bitmap> saving_bitmaps;
    public final static int savings = 10;
    public final static int capacity = 100;

    // enabled method for preprocess
    // 0 : getframeatindex at runtime, 1: get allframes at once, 2: get chunks if needed
    public final static int preprocess_runtime = 0;
    public final static int preprocess_allframe = 1;
    public final static int preprocess_chunks = 2;
    public final static int preprocessing_way = preprocess_runtime;


    public FrameManager(MediaMetadataRetriever m) {
        this.m = m;
        this.stateManager = new StateManager();
        frm_idx = 0;
        all_bitmaps = new ArrayList<>();
        saving_bitmaps = new ArrayList<>();
    }

    /**
     * Preprocess
     *  - All at once. if your device's RAM is enough to save all bitmaps in Video, this method can be used.
     */
    public void preprocess() {
        if (preprocessing_way == preprocess_allframe) {
            // all frame should be fine at recent versions of phones, but if ram is not enough, then this should be optimized.
            int frm_cnt = Integer.parseInt(m.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT));
            all_bitmaps.addAll(m.getFramesAtIndex(0, frm_cnt));
        } else if (preprocessing_way == preprocess_chunks)
            preprocess_recursive();
    }

    /**
     * Preprocess (recursive version)
     * if your memory is not enough to save all bitmaps in video, this method has to be used.
     * default saved bitmaps : 100
     */
    public void preprocess_recursive() {
        int frm_cnt = Integer.parseInt(m.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT));
        // removing session
        if (frm_saved > 0) {
            saving_bitmaps.clear();
            for (int i = 0; i < savings; i++) {
                Bitmap tmp_bitmap = Bitmap.createBitmap(all_bitmaps.get(i));
                saving_bitmaps.add(tmp_bitmap);
            }
            all_bitmaps.clear();
        }
        // adding session
        if (frm_saved < frm_cnt / capacity) {
//            Log.v(TAG, "Yousung [add   ]: i : " + i + " test_list length : " + test_list.size());
            all_bitmaps.addAll(m.getFramesAtIndex(frm_saved++ * capacity, capacity));
        } else {
//            Log.v(TAG, "Yousung [add   ]: i : " + i + " test_list length : " + test_list.size());
            all_bitmaps.addAll(m.getFramesAtIndex(frm_saved++ * capacity, frm_cnt % capacity));
        }
    }

    /**
     * [211112] works to do
     * if preprocessing way == preprocess_chunks, call preprocess_recursive in process to save images.
     */
    public Output_Data process(int found_status, boolean need_rewind, int rewind) {
        Log.d(TAG, "[status, need_rewind, rewind_param] [" + found_status + "," + need_rewind + "," + rewind + "]");
        Log.d(TAG, "[current state, maintain, count : " + this.stateManager.current_state.state + ", " + this.stateManager.current_state.maintain + ", " + this.stateManager.current_state.count);
        Log.v(TAG, "frm_idx for debug" + frm_idx);

        if (need_rewind) {
            if (rewind == this.stateManager.Initial_rewind - 1) {
                frm_idx -= rewind;
                frm_idx = frm_idx >= 0 ? frm_idx : 0;
            } else
                frm_idx += 1;
        } else {
            this.stateManager.switch_state((found_status > 0));

            // if sampling has done at proper index.
            while (true) {
                frm_idx += 1;
                if (this.stateManager.validation())
                    break;
            }
        }
        Log.d(TAG, "frm_idx : " + frm_idx);

        /**
         * There are 3 ways to get frames
         * 1) get frame at index through MetadataRetriever : gets bitmap at specific index at runtime. this works but slow : (takes more than 100ms per 1 image)
         * 2) save all frames at once through 'preprocess', then pick one at it : this works fast (takes about 20ms per 1 image), but lots of RAM needed.
         * 3) save chunks of images (per 100 images) in list through 'preprocess_recursive' , and pick up if rewind needs.
         *
         * to work in every platform, we pick 3) choice to work within it.
         */
        Bitmap sampled_bitmap;
        if (preprocessing_way == preprocess_allframe)
            sampled_bitmap = all_bitmaps.get(frm_idx);
        else if (preprocessing_way == preprocess_runtime)
            sampled_bitmap = m.getFrameAtIndex(frm_idx);

        return new Output_Data(sampled_bitmap, frm_idx);
    }

    /**
     * Frame manager needs state manager which can points out what current state is. (used in sampling state)
     * So we add state manager inside of frame manager to manage sampling state.
     * */
    public class StateManager {
        State current_state;

        // values which doens't change
        final int Initial_state = 0;
        final int Detection_state = 1;
        final int Stabilize_state = 2;

        // values that can be changed for test
        int Initial_sampling = 5;
        int Initial_maintain = 0;
        int Initial_rewind = Initial_sampling;
        int Detection_maintain = 3;
        int Detection_sampling = 1;
        int Stabilize_sampling = 2;

        public class State {
            int state;
            int maintain;
            int count;
            public State () {
                this.state = Initial_state;
                this.maintain = Initial_maintain;
                this.count = 4;
            }
        }

        public StateManager() {
            current_state = new State();
        }

        public void up_state() {
            if (current_state.state != 2) {
                current_state.state += 1;
                current_state.maintain = 0;
                current_state.count = 0;
            }
        }

        public void down_state() {
            if (current_state.state != 0) {
                current_state.state -= 1;
                current_state.maintain = 0;
                current_state.count = 0;
            }
        }

        public void switch_state(boolean valid) {
            Integer state = current_state.state;
            if (valid) {
                if (state == Initial_state)
                    up_state();
                else if (state == Detection_state) {
                    current_state.maintain += 1;
                    if (current_state.maintain == Detection_maintain)
                        up_state();
                }
            } else {
                if (state > Initial_state)
                    down_state();
            }
        }

        public boolean validation() {
            current_state.count += 1;
            Integer state = current_state.state;
            if (state == Initial_state)
                return (current_state.count % Initial_sampling == 0);
            else if (state == Detection_state)
                return true;
            else if (state == Stabilize_state)
                return (current_state.count % Stabilize_sampling == 0);
            else
                Log.v(TAG, "weird state is shown [" + state + "]");
                return false;
        }
    }
}
