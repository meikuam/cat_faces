package com.meikuam.cat_faces;

import androidx.appcompat.app.AppCompatActivity;

import android.util.Log;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuffXfermode;
import android.graphics.PorterDuff;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;


public class MainActivity extends AppCompatActivity {
    private static int RESULT_LOAD_IMAGE = 1;
    private static Module module = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]  {android.Manifest.permission.READ_EXTERNAL_STORAGE, android.Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        }



        Button buttonLoadImage = (Button) findViewById(R.id.button);
        Button detectButton = (Button) findViewById(R.id.detect);


        buttonLoadImage.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View arg0) {
                Intent i = new Intent(
                        Intent.ACTION_PICK,
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

                startActivityForResult(i, RESULT_LOAD_IMAGE);


            }
        });
        detectButton.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View arg0) {
                Log.d("cat_faces", "start of prediction");
                Bitmap bitmap = null;
//                bitmap.recycle();
//                bitmap = null;

                // Loading the model file.
                if (module == null) {
                    try {
                        module = Module.load(getModelPath(MainActivity.this, "traced_model.pt"));
                    } catch (IOException e) {
                        finish();
                    }
                }
                int model_input_width = 416;
                int model_input_height = 416;

                // prepare image to be input of model

                //Getting the image from the image view
                ImageView imageView = (ImageView) findViewById(R.id.image);
                //Read the image as Bitmap
                bitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
                Log.d("cat_faces", "open image");


                int image_width = bitmap.getWidth();
                int image_height = bitmap.getHeight();

                int padding_top = 0;
                int padding_left = 0;
                int padding_right = 0;
                int padding_bottom = 0;

                if (image_height > image_width) {
                    padding_bottom = padding_top = 0;
                    padding_left = (image_height - image_width) / 2;
                    padding_right = image_height - (image_width + padding_left);
                } else {
                    padding_left = padding_right = 0;
                    padding_top = (image_width - image_height) / 2;
                    padding_bottom = image_width - (image_height + padding_top);
                }
                Bitmap input_bitmap = padImage(bitmap, padding_left, padding_top, padding_right,  padding_bottom);
                Log.d("cat_faces", "pad image");
                input_bitmap = resizeImage(input_bitmap, model_input_width, model_input_height);
                Log.d("cat_faces", "resize image");

                //Input Tensor
                final Tensor input_tensor = TensorImageUtils.bitmapToFloat32Tensor(
                        input_bitmap,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                        TensorImageUtils.TORCHVISION_NORM_STD_RGB
                );
                input_bitmap.recycle();
                input_bitmap = null;
                Log.d("cat_faces", "to tensor image");

                //Calling the forward of the model to run our input
                final Tensor output_tensor = module.forward(IValue.from(input_tensor)).toTensor();
                Log.d("cat_faces", "forward network");

                // post process
                float[] float_data = output_tensor.getDataAsFloatArray(); // 3 x width x height
                Bitmap output_bitmap = Bitmap.createBitmap(model_input_width, model_input_height, Bitmap.Config.ARGB_8888);
                int offset = model_input_height * model_input_width;
                for (int i = 0; i < model_input_width; i++) {
                    for  (int j = 0; j < model_input_height; j++) {
                        int argb_color = 0;
                        if (float_data[offset + model_input_width * j + i] > 0.5) {
                            argb_color = Color.argb(255, 255, 255, 255);
                        } else {
                            argb_color = Color.argb(0, 0, 0, 0);
                        }
                        output_bitmap.setPixel(i, j, argb_color);
                    }
                }
                Log.d("cat_faces", "fill output bitmap image");
                output_bitmap = resizeAndCropImage(
                        output_bitmap,
                        image_width + padding_left + padding_right,
                        image_height + padding_bottom + padding_top,
                        padding_left,
                        padding_top,
                        image_width,
                        image_height
                );
                Log.d("cat_faces", "resize and crop output image");
//                output_bitmap = resizeImage(
//                        output_bitmap,
//                        image_width + padding_left + padding_right,
//                        image_height + padding_bottom + padding_top
//                );
//                Log.d("cat_faces", "resize output image");
//                output_bitmap = cropImage(
//                        output_bitmap,
//                        padding_left,
//                        padding_top,
//                        image_width,
//                        image_height
//                );
                Log.d("cat_faces", "crop output image");

                bitmap = maskImage(bitmap, output_bitmap);
                Log.d("cat_faces", "mask image");
                imageView.setImageBitmap(bitmap);
                Log.d("cat_faces", "set bitmap image");

            }
        });

    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        //This functions return the selected image from gallery
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();
            String[] filePathColumn = { MediaStore.Images.Media.DATA };

            Cursor cursor = getContentResolver().query(selectedImage,
                    filePathColumn, null, null, null);
            cursor.moveToFirst();

            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();

            ImageView imageView = (ImageView) findViewById(R.id.image);
            imageView.setImageBitmap(BitmapFactory.decodeFile(picturePath));

            //Setting the URI so we can read the Bitmap from the image
            imageView.setImageURI(null);
            imageView.setImageURI(selectedImage);


        }


    }

    public static String getModelPath(Context context, String modelName) throws IOException {
        File file = new File(context.getFilesDir(), modelName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(modelName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    public Bitmap padImage(Bitmap original, int padding_left, int padding_top, int padding_right, int padding_bottom) {
        Bitmap outputimage = Bitmap.createBitmap(
                original.getWidth() + padding_left + padding_right,
                original.getHeight() + padding_top + padding_bottom,
                Bitmap.Config.ARGB_8888
        );
        Canvas can = new Canvas(outputimage);
        can.drawARGB(255,128,128, 128);
        can.drawBitmap(original, padding_left, padding_top, null);
        return outputimage;
    }

    public Bitmap cropImage(Bitmap original, int x, int y, int width, int height) {
        x = x > 0 ? x : 0;
        y = y > 0 ? y : 0;
        width = original.getWidth() - x > width ? width : original.getWidth() - x;
        height = original.getHeight() - y > height ? height : original.getHeight() - y;
        return Bitmap.createBitmap(
                original,
                x,
                y,
                width,
                height
        );
    }

    public Bitmap resizeImage(Bitmap original, int width, int height) {
        return Bitmap.createScaledBitmap(original, width, height, false);
    }

    public Bitmap resizeAndCropImage(Bitmap original, int new_width, int new_height, int x, int y, int crop_width, int crop_height) {
        // resize original image then crop,
        // but the trick is that we get original image crop it, then resize to save memory
        int orig_width = original.getWidth();
        int orig_height = original.getHeight();
        float[] scales = {new_width/orig_width, new_height/orig_height};
        int scaled_x = (int) (x * scales[0]);
        int scaled_y = (int) (x * scales[1]);
        int scaled_crop_width = (int) (crop_width * scales[0]);
        int scaled_crop_height = (int) (crop_height * scales[1]);


        scaled_x = scaled_x > 0 ? scaled_x : 0;
        scaled_y = scaled_y > 0 ? scaled_y : 0;
        scaled_crop_width = new_width - scaled_x > scaled_crop_width ? scaled_crop_width : new_width - scaled_x;
        scaled_crop_height = new_height - scaled_y > scaled_crop_height ? scaled_crop_height : new_height - scaled_y;


        return Bitmap.createScaledBitmap(
                Bitmap.createBitmap(
                        original,
                        scaled_x,
                        scaled_y,
                        scaled_crop_width,
                        scaled_crop_height
                ), new_width, new_width, false);
    }


    private Bitmap maskImage(Bitmap original, Bitmap mask) {
        Bitmap result = null;
        try {
            if (original != null)
            {
                int intWidth = original.getWidth();
                int intHeight = original.getHeight();
                result = Bitmap.createBitmap(original.getWidth(), original.getHeight(), Bitmap.Config.ARGB_8888);
                Bitmap scaled_mask = Bitmap.createScaledBitmap(mask, intWidth, intHeight, false);
                Canvas mCanvas = new Canvas(result);
                Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
                paint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.DST_IN));
                mCanvas.drawBitmap(original, 0, 0, null);
                mCanvas.drawBitmap(scaled_mask, 0, 0, paint);
                paint.setXfermode(null);
                paint.setStyle(Paint.Style.STROKE);
            }
        } catch (OutOfMemoryError o) {
            o.printStackTrace();
        }
        return result;
    }


//    public void saveTempBitmap(Bitmap bitmap) {
//        if (isExternalStorageWritable()) {
//            saveImage(bitmap);
//        }else{
//            //prompt the user or do something
//        }
//    }

//    private void saveImage(Bitmap finalBitmap) {
//
//        String root = Environment.getExternalStorageDirectory().toString();
//        File myDir = new File(root + "/saved_images");
//        myDir.mkdirs();
//
//        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
//        String fname = "Shutta_"+ timeStamp +".jpg";
//
//        File file = new File(myDir, fname);
//        if (file.exists()) file.delete ();
//        try {
//            FileOutputStream out = new FileOutputStream(file);
//            finalBitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
//            out.flush();
//            out.close();
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//    }
//
//    /* Checks if external storage is available for read and write */
//    public boolean isExternalStorageWritable() {
//        String state = Environment.getExternalStorageState();
//        if (Environment.MEDIA_MOUNTED.equals(state)) {
//            return true;
//        }
//        return false;
//    }
}
