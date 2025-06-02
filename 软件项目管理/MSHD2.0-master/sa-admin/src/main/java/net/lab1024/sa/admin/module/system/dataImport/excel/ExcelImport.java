package net.lab1024.sa.admin.module.system.dataImport.excel;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.CellType;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

public class ExcelImport {
    public JSONObject readUsersExcel(String fileName, String sheetName) throws JSONException, IOException {
        try (BufferedInputStream inputStream = new BufferedInputStream(Files.newInputStream(Paths.get(fileName)));
             XSSFWorkbook workbook = new XSSFWorkbook(inputStream)) {
            // 选择要处理的sheet名称
            XSSFSheet sheet = workbook.getSheet(sheetName);
            // 处理表头
            JSONArray jsonArray = processHeader(sheet);
            // 处理行数据
            JSONArray jsonArray_row = processRows(sheet, jsonArray);
            // Convert JSON Array to JSON Object if needed
            JSONObject jsonObject = new JSONObject();
            jsonObject.put(sheetName, jsonArray_row);
            return jsonObject;
        }
    }

    // 处理表头
    private JSONArray processHeader(XSSFSheet sheet) {
        XSSFRow head = sheet.getRow(0);
        JSONArray jsonArray = new JSONArray();
        for (int cellNum = 0; cellNum < head.getPhysicalNumberOfCells(); cellNum++) {
            String columnName = head.getCell(cellNum).getStringCellValue();
            jsonArray.put(columnName);
        }
        return jsonArray;
    }

    // 处理行
    private JSONArray processRows(XSSFSheet sheet, JSONArray headerArray) throws JSONException {
        JSONArray jsonArray = new JSONArray();
        int numberOfCells = headerArray.length(); // 保存列数
        for (int rowNum = 1; rowNum < sheet.getPhysicalNumberOfRows(); rowNum++) {
            XSSFRow row = sheet.getRow(rowNum);
            JSONObject rowObject = new JSONObject();
            for (int cellNum = 0; cellNum < numberOfCells; cellNum++) { // 使用保存的列数
                if (row.getCell(cellNum) != null) {
                    String columnName = headerArray.getString(cellNum);
                    String cellValue = getCellValueAsString(row.getCell(cellNum));
                    rowObject.put(columnName, cellValue);
                }
            }
            jsonArray.put(rowObject);
        }
        // 打印处理后的 JSON 数据
        return jsonArray;
    }

    private String getCellValueAsString(Cell cell) {
        if (cell.getCellType() == CellType.NUMERIC) {
            return String.valueOf((int) cell.getNumericCellValue());
        } else {
            return cell.getStringCellValue();
        }
    }
}