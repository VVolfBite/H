package main;

import input.*;
import output.*;
import shift.*;
import sort.*;
import process.*;

import java.util.*;
import java.io.*;

public class KWICSystem {
    private final InputHandler inputHandler;
    private final ProcessStrategy processStrategy;
    private final ShiftGenerator shiftGenerator;
    private final SortStrategy sortStrategy;
    private final OutputHandler outputHandler;
    

    /**
     * 构造函数
     * @param inputHandler 输入处理模块
     * @param processStrategy 数据处理模块
     * @param shiftGenerator 循环位移生成模块
     * @param sortStrategy 排序模块
     * @param outputHandler 输出处理模块
     */
    public KWICSystem(InputHandler inputHandler,ProcessStrategy processStrategy, ShiftGenerator shiftGenerator,
                      SortStrategy sortStrategy, OutputHandler outputHandler) {
        this.inputHandler = inputHandler;
        this.processStrategy = processStrategy;
        this.shiftGenerator = shiftGenerator;
        this.sortStrategy = sortStrategy;
        this.outputHandler = outputHandler;
        
    }

    /**
     * 执行 KWIC 流程
     * @throws IOException 如果输入或输出过程中发生错误
     */
    public void execute() throws IOException {
        // Step 1: 输入
        List<String> lines = inputHandler.readLines();

        // Step 1* : 拓展处理输入
        lines = processStrategy.processData(lines);

        // Step 2: 生成循环位移
        List<String> allShifts = new ArrayList<>();
        for (String line : lines) {
            allShifts.addAll(shiftGenerator.generateShifts(line));
        }

        // Step 3: 排序
        List<String> sortedShifts = sortStrategy.sort(allShifts);

        // Step 4: 输出
        outputHandler.writeLines(sortedShifts);
    }

    /**
     * 主函数入口
     * @param args 命令行参数
     *             args[0]: 输入类型 ("file" 或 "socket")
     *             args[1]: 输出文件路径
     *             args[2]: 输入源 (文件路径或端口号)
     * @throws IOException 如果输入或输出过程中发生错误
     */
    public static void main(String[] args) throws IOException {
        if (args.length < 3) {
            System.out.println("Usage: java KWICSystem <inputType> <outputFilePath> <inputSource>");
            System.out.println("Example for file input: java KWICSystem file output.txt input.txt");
            System.out.println("Example for socket input: java KWICSystem socket output.txt 8080");
            return;
        }

        // 配置输入处理
        String inputType = args[0];
        InputHandler inputHandler;
        if (inputType.equals("file")) {
            inputHandler = new FileInputHandler(args[2]);
        } else if (inputType.equals("socket")) {
            inputHandler = new SocketInputHandler(Integer.parseInt(args[2]));
        } else {
            throw new IllegalArgumentException("Invalid input type: " + inputType);
        }

        // 配置其他模块
        ShiftGenerator shiftGenerator = new CyclicShiftGenerator();
        SortStrategy sortStrategy = new DefaultSortStrategy(); // 默认字典序排序
        ProcessStrategy processStrategy = new DefaultProcessStrategy();
        OutputHandler outputHandler = new FileOutputHandler(args[1]);

        // 创建系统实例并执行
        KWICSystem kwic = new KWICSystem(inputHandler,processStrategy, shiftGenerator, sortStrategy, outputHandler);
        kwic.execute();
    }
}
