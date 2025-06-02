package net.lab1024.sa.admin.module.system.earthquake.controller;

import cn.hutool.extra.servlet.ServletUtil;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import net.lab1024.sa.admin.common.AdminBaseController;
import net.lab1024.sa.admin.constant.AdminSwaggerTagConst;
import net.lab1024.sa.admin.module.system.earthquake.domain.form.EarthquakeAddForm;
import net.lab1024.sa.admin.module.system.earthquake.domain.form.EarthquakeQueryForm;
import net.lab1024.sa.admin.module.system.earthquake.domain.vo.EarthquakeVO;
import net.lab1024.sa.admin.module.system.earthquake.service.EarthquakeService;
import net.lab1024.sa.common.common.constant.RequestHeaderConst;
import net.lab1024.sa.common.common.domain.PageResult;
import net.lab1024.sa.common.common.domain.ResponseDTO;
import net.lab1024.sa.common.module.support.operatelog.annoation.OperateLog;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import javax.validation.Valid;
import java.util.List;

/**
 * 员工
 *
 * @Author 1024创新实验室-主任: 卓大
 * @Date 2021-12-09 22:57:49
 * @Wechat zhuoda1024
 * @Email lab1024@163.com
 * @Copyright 1024创新实验室 （ https://1024lab.net ）
 */
@RestController
@OperateLog
@Api(tags = {AdminSwaggerTagConst.System.SYSTEM_EARTHQUAKE})
public class EarthquakeController extends AdminBaseController {

    @Autowired
    private EarthquakeService earthquakeService;

    @PostMapping("/earthquake/query")
    @ApiOperation(value = "震情管理查询 @author 卓大")
    public ResponseDTO<PageResult<EarthquakeVO>> query(@Valid @RequestBody EarthquakeQueryForm query) {
        return earthquakeService.queryEarthquake(query);
    }

    @ApiOperation(value = "添加震情信息 @author 卓大")
    @PostMapping("/earthquake/add")
//    @PreAuthorize("@saAuth.checkPermission('system:employee:add')")
    public ResponseDTO<String> addEarthquake(@Valid @RequestBody EarthquakeAddForm earthquakeAddForm) {
        return earthquakeService.addEarthquake(earthquakeAddForm);
    }

    @ApiOperation(value = "更新震情信息 @author 卓大")
    @PostMapping("/earthquake/update")
//    @PreAuthorize("@saAuth.checkPermission('system:employee:update')")
    public ResponseDTO<String> updateSupportDisaster(@Valid @RequestBody EarthquakeAddForm earthquakeUpdateForm) {
        return earthquakeService.updateEarthquake(earthquakeUpdateForm);
    }

    @ApiOperation(value = "批量删除员工 @author 卓大")
    @PostMapping("/earthquake/update/batch/delete")
//    @PreAuthorize("@saAuth.checkPermission('system:employee:delete')")
    public ResponseDTO<String> batchUpdateDeleteFlag(@RequestBody List<String> earthquakeCodeList) {
        return earthquakeService.batchUpdateDeleteFlag(earthquakeCodeList);
    }

    @ApiOperation(value = "批量增加员工 @author 卓大")
    @PostMapping("/earthquake/batch/add")
//    @PreAuthorize("@saAuth.checkPermission('system:employee:delete')")
    public ResponseDTO<String> batchAddEarthquake(@RequestBody String fileKey) {
        return earthquakeService.batchAddEarthquake(fileKey);    }

    @ApiOperation("查询所有震情信息 @author 卓大")
    @GetMapping("/earthquake/queryAll")
    public ResponseDTO<List<EarthquakeVO>> queryAllEarthquake() {
        return earthquakeService.queryAllEarthquake();
    }

}

