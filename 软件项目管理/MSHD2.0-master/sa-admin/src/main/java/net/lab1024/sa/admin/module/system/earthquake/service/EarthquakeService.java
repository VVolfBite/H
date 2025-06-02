package net.lab1024.sa.admin.module.system.earthquake.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.Lists;
import net.lab1024.sa.admin.module.system.dataImport.excel.ExcelImport;
import net.lab1024.sa.admin.module.system.earthquake.dao.EarthquakeDao;
import net.lab1024.sa.admin.module.system.earthquake.domain.entity.EarthquakeEntity;
import net.lab1024.sa.admin.module.system.earthquake.domain.form.EarthquakeAddForm;
import net.lab1024.sa.admin.module.system.earthquake.domain.form.EarthquakeQueryForm;
import net.lab1024.sa.admin.module.system.earthquake.domain.vo.EarthquakeVO;
import net.lab1024.sa.admin.module.system.earthquake.manager.EarthquakeManager;
import net.lab1024.sa.common.common.code.UserErrorCode;
import net.lab1024.sa.common.common.domain.PageResult;
import net.lab1024.sa.common.common.domain.ResponseDTO;
import net.lab1024.sa.common.common.util.SmartBeanUtil;
import net.lab1024.sa.common.common.util.SmartPageUtil;
import net.lab1024.sa.common.module.support.file.dao.FileDao;
import net.lab1024.sa.common.module.support.file.domain.vo.FileVO;
import net.lab1024.sa.common.module.support.token.TokenService;
import org.apache.commons.codec.digest.DigestUtils;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.RandomStringUtils;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class EarthquakeService {

    @Value("${file.storage.local.path}")
    private String localPath;

    @Autowired
    private FileDao fileDao;

    @Autowired
    private EarthquakeDao earthquakeDao;

    @Autowired
    private EarthquakeManager earthquakeManager;

    @Autowired
    private TokenService tokenService;

    private static final ObjectMapper objectMapper = new ObjectMapper();


    public EarthquakeVO getEarthquakeByCode(String code) {
        return earthquakeDao.getEarthquakeByCode(code);
    }


    /**
     * 查询震情列表
     *
     * @param earthquakeQueryForm
     * @return
     */
    public ResponseDTO<PageResult<EarthquakeVO>> queryEarthquake(EarthquakeQueryForm earthquakeQueryForm) {
        earthquakeQueryForm.setDeletedFlag(false);
        Page pageParam = SmartPageUtil.convert2PageQuery(earthquakeQueryForm);

        List<EarthquakeVO> earthquakeList = earthquakeDao.queryEarthquake(pageParam, earthquakeQueryForm);
        System.out.println(earthquakeList);
        if (CollectionUtils.isEmpty(earthquakeList)) {
            PageResult<EarthquakeVO> PageResult = SmartPageUtil.convert2PageResult(pageParam, earthquakeList);
            return ResponseDTO.ok(PageResult);
        }

        List<String> earthquakeCodeList = earthquakeList.stream().map(EarthquakeVO::getCode).collect(Collectors.toList());

        PageResult<EarthquakeVO> PageResult = SmartPageUtil.convert2PageResult(pageParam, earthquakeList);
        return ResponseDTO.ok(PageResult);
    }

    /**
     * 批量新增震情
     */
    public synchronized ResponseDTO<String> batchAddEarthquake(String fileKey) {
        System.out.println(fileKey);
        String tempString =fileKey.replaceAll("%2F", "/");
        fileKey=tempString.replaceAll("=$", "");
        String filePath = localPath + fileKey;
        System.out.println(filePath);
        Path path = Paths.get(filePath);
        if (Files.exists(path)) {
            System.out.println("File exists!");
        } else {
            System.out.println("File does not exist!");
        }
        ExcelImport excelImport = new ExcelImport();
        try {
            JSONObject check = excelImport.readUsersExcel(filePath, "sheet1");
            // 获取 "sheet1" 对应的 JSONArray
            JSONArray sheet1Array = check.getJSONArray("sheet1");
            System.out.println(sheet1Array);
            // 遍历 JSONArray 中的每个 JSONObject，获取 "test" 值
            for (int i = 0; i < sheet1Array.length(); i++) {
                JSONObject item = sheet1Array.getJSONObject(i);
                if (item.has("code")){
                    String code = item.getString("code");
                    EarthquakeAddForm earthquakeAddForm = new EarthquakeAddForm();
                    System.out.println(code);
                    earthquakeAddForm.setCode(code);
                    System.out.println(earthquakeAddForm);
                    addEarthquake(earthquakeAddForm);
                }
                else
                    break;
            }
        } catch (IOException | JSONException Exception) {
            Exception.printStackTrace();
        }
        return ResponseDTO.ok();
    }

    /**
     * 新增震情
     *
     * @param earthquakeAddForm
     * @return
     */
    public synchronized ResponseDTO<String> addEarthquake(EarthquakeAddForm earthquakeAddForm) {
        // 校验名称是否重复
        EarthquakeEntity earthquakeEntity = earthquakeDao.selectById(earthquakeAddForm.getCode());
        System.out.println(earthquakeEntity);
        if (null != earthquakeEntity) {
            if (earthquakeEntity.getDeletedFlag() == Boolean.FALSE) {
                return ResponseDTO.userErrorParam("震情码重复");
            } else {
                EarthquakeEntity entity = SmartBeanUtil.copy(earthquakeAddForm, EarthquakeEntity.class);

                System.out.println(entity);
                // 保存数据
                entity.setDeletedFlag(Boolean.FALSE);
                earthquakeManager.updateEarthquake(entity);
            }
        } else {
            String code = earthquakeAddForm.getCode();
            String geoCode = code.substring(0, 12);
            String time = code.substring(12, 26);
            String sourceCode = code.substring(26, 27);
            String subsourceCode = code.substring(27, 29);
            String carrierCode = code.substring(29, 30);
            String disasterCode = code.substring(30, 31);
            String subdisasterCode = code.substring(31, 33);
            String degreeCode = code.substring(33, 36);

            //解码地址信息
            String location = earthquakeDao.getByGeoCode(geoCode);
            earthquakeAddForm.setLocation(location);
            System.out.println(location);

            //解码时间信息
            SimpleDateFormat inputFormat = new SimpleDateFormat("yyyyMMddHHmmss");
            SimpleDateFormat outputFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

            try {
                // 将字符串解析为日期时间对象
                java.util.Date date = inputFormat.parse(time);
                // 将日期时间对象格式化为指定格式的字符串
                String output = outputFormat.format(date);
                earthquakeAddForm.setDatetime(output);
                System.out.println(output);
            } catch (ParseException e) {
                e.printStackTrace();
            }

            //解码剩下的灾情信息
            try {
                File file1 = new File("sa-admin/src/main/java/net/lab1024/sa/admin/module/system/earthquake/decoder.json");
                File file2 = new File("sa-admin/src/main/java/net/lab1024/sa/admin/module/system/earthquake/subdecoder.json");
                Map<String, Map<String, String>> mappingData = objectMapper.readValue(file1, Map.class);
                Map<String, Map<String, Map<String, String>>> mappingSubData = objectMapper.readValue(file2, Map.class);

                String source = mappingData.get("sourceMap").get(sourceCode);
                earthquakeAddForm.setSource(source);

                String subsource = mappingSubData.get("subsourceMap").get(sourceCode).get(subsourceCode);
                earthquakeAddForm.setSubsource(subsource);

                String carrier = mappingData.get("carrierMap").get(carrierCode);
                earthquakeAddForm.setCarrier(carrier);

                String disaster = mappingData.get("disasterMap").get(disasterCode);
                earthquakeAddForm.setDisaster(disaster);

                String subdisaster = mappingSubData.get("subdisasterMap").get(disasterCode).get(subdisasterCode);
                earthquakeAddForm.setSubdisaster(subdisaster);

                String degree = mappingSubData.get("degreeMap").get(disasterCode).get(degreeCode.substring(0, 3));
                earthquakeAddForm.setDegree(degree);

            } catch (Exception e) {
                e.printStackTrace();
            }

            System.out.println(earthquakeAddForm);


            EarthquakeEntity entity = SmartBeanUtil.copy(earthquakeAddForm, EarthquakeEntity.class);

            System.out.println(entity);
            // 保存数据
            entity.setDeletedFlag(Boolean.FALSE);
            earthquakeManager.saveEarthquake(entity);
        }
        return ResponseDTO.ok();
    }

    /**
     * 更新震情信息
     *
     * @param earthquakeUpdateForm
     * @return
     */
    public synchronized ResponseDTO<String> updateEarthquake(EarthquakeAddForm earthquakeUpdateForm) {

        String code = earthquakeUpdateForm.getCode();
        EarthquakeEntity earthquakeEntity = earthquakeDao.selectById(code);
        if (null == earthquakeEntity) {
            return ResponseDTO.error(UserErrorCode.DATA_NOT_EXIST);
        }

        EarthquakeEntity entity = SmartBeanUtil.copy(earthquakeUpdateForm, EarthquakeEntity.class);

        // 更新数据
        earthquakeManager.updateEarthquake(entity);

        return ResponseDTO.ok();
    }

    /**
     * 批量删除员工
     *
     * @param earthquakeCodeList 震情码列表
     * @return
     */
    public ResponseDTO<String> batchUpdateDeleteFlag(List<String> earthquakeCodeList) {
        if (CollectionUtils.isEmpty(earthquakeCodeList)) {
            return ResponseDTO.ok();
        }
        List<EarthquakeEntity> earthquakeEntityList = earthquakeManager.listByIds(earthquakeCodeList);
        if (CollectionUtils.isEmpty(earthquakeEntityList)) {
            return ResponseDTO.ok();
        }
        // 更新删除
        List<EarthquakeEntity> deleteList = earthquakeCodeList.stream().map(e -> {
            EarthquakeEntity updateEarthquake = new EarthquakeEntity();
            updateEarthquake.setCode(e);
            updateEarthquake.setDeletedFlag(true);
            return updateEarthquake;
        }).collect(Collectors.toList());
        earthquakeManager.updateBatchById(deleteList);

        return ResponseDTO.ok();
    }


    /**
     * 查询全部震情信息
     *
     * @return
     */
    public ResponseDTO<List<EarthquakeVO>> queryAllEarthquake() {
        List<EarthquakeVO> earthquakeList = earthquakeDao.selectEarthquakeByDeleted(Boolean.FALSE);
        System.out.println(earthquakeList);
        return ResponseDTO.ok(earthquakeList);
    }

}
