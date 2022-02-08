package com.ml_gram.project_coordination_final;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Color;
import android.graphics.Point;
import android.graphics.RectF;
import android.os.Bundle;
import android.util.Log;
import android.view.Display;
import android.view.View;
import android.widget.Button;

import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet;

import java.util.ArrayList;

public class GraphActivity extends AppCompatActivity {
    private final String TAG = "Graph Activity";
    private LineChart lineChart;
    private Button btn_back;
    private ArrayList<RectF> coords_results;
    private ArrayList<RectF> coords_modified_results;
    private ArrayList<Integer> state_results;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_graph);

        lineChart = findViewById(R.id.linechart);
        btn_back = findViewById(R.id.button_back);

        coords_results = new ArrayList<>();
        state_results = new ArrayList<>();
        coords_modified_results = new ArrayList<>();

        Intent intent = getIntent();
        coords_results = intent.getParcelableArrayListExtra("coords_results");
        coords_modified_results = intent.getParcelableArrayListExtra("coords_modified_results");
        state_results = intent.getIntegerArrayListExtra("state_results");

//        Log.v(TAG, "coords_results = " + coords_results.toString());

        // [Yousung] add location information to transform coordination, but it needs to be counted in main activity
        Display display = getWindowManager().getDefaultDisplay();
        Point size = new Point();
        display.getRealSize(size);
        Log.v(TAG, "width : " + size.x + " height : " + size.y);

        ArrayList<Entry> values = new ArrayList<>();
        ArrayList<Entry> modified_values = new ArrayList<>();
        ArrayList<Entry> values_state = new ArrayList<>();

        for (int i = 0; i < coords_results.size(); i++) {
            try {
                float val = 0;
                int val_state = 0;
                if (coords_results.get(i).centerX() > 0)
                    val = (coords_results.get(i).centerX() * size.x / 640) + (coords_results.get(i).centerY() * size.y / 640);
                else
                    val = coords_results.get(i).left;
                val_state = state_results.get(i);

                values.add(new Entry(i, val));
                values_state.add(new Entry(i, val_state));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        for (int i = 0; i < coords_modified_results.size(); i++) {
            try {
                float val = 0;
                if (coords_modified_results.get(i).centerX() > 0)
                    val = (coords_modified_results.get(i).centerX() * size.x / 640) + (coords_modified_results.get(i).centerY() * size.y / 640);
                else
                    val = coords_modified_results.get(i).left;

                modified_values.add(new Entry(i, val));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        LineDataSet set1, set2, set3;
        set1 = new LineDataSet(values, "Cursor Coordination / frames");
        set2 = new LineDataSet(values_state, "State / frames");
        set3 = new LineDataSet(modified_values, "Cursor modified coordination / frames");

        ArrayList<ILineDataSet> dataSets = new ArrayList<>();
        dataSets.add(set1); // add the data sets
        dataSets.add(set2);
        dataSets.add(set3);

        // create a data object with the data sets
        LineData data = new LineData(dataSets);

        // black lines and points
        set1.setColor(Color.BLACK);
        set1.setCircleColor(Color.BLACK);

        set2.setColor(Color.RED);
        set2.setCircleColor(Color.RED);

        set3.setColor(Color.BLUE);
        set3.setCircleColor(Color.BLUE);

        // set data
        lineChart.setData(data);
        lineChart.setDescription(null);

        lineChart.invalidate();

        btn_back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish();
            }
        });
    }


}