package edu.cmu.ml.rtw.pra.experiments;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import edu.cmu.ml.rtw.users.matt.util.Dictionary;

public class DatasetFactory {
  public Dataset fromFile(String path, Dictionary dict) throws IOException {
    return fromReader(new BufferedReader(new FileReader(path)), dict);
  }

  public Dataset fromReader(BufferedReader reader, Dictionary dict) throws IOException {
    return Dataset.readFromReader(reader, dict);
  }
}
