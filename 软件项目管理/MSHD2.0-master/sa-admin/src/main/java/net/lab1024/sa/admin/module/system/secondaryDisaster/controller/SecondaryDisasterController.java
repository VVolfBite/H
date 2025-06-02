package net.lab1024.sa.admin.module.system.secondaryDisaster.controller;

import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import net.lab1024.sa.admin.common.AdminBaseController;
import net.lab1024.sa.admin.constant.AdminSwaggerTagConst;

import net.lab1024.sa.admin.module.system.secondaryDisaster.domain.form.SecondaryDisasterAddForm;
import net.lab1024.sa.admin.module.system.secondaryDisaster.domain.form.SecondaryDisasterQueryForm;
import net.lab1024.sa.admin.module.system.secondaryDisaster.domain.vo.SecondaryDisasterVO;
import net.lab1024.sa.admin.module.system.secondaryDisaster.service.SecondaryDisasterService;
import net.lab1024.sa.common.common.domain.PageResult;
import net.lab1024.sa.common.common.domain.ResponseDTO;

import net.lab1024.sa.common.module.support.operatelog.annoation.OperateLog;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import javax.validation.Valid;
import java.util.List;

/**
 * ClassName: SecondaryDisasterController
 * Package: net.lab1024.sa.admin.module.system.secondaryDisaster.controller
 * Description:
 *
 * @Author 幻秋
 * @Create 2023/11/5  20:38
 * @Version 1.0
 */
@RestController
@OperateLog
@Api(tags = {AdminSwaggerTagConst.System.SYSTEM_SECONDARY_DISASTER})
public class SecondaryDisasterController extends AdminBaseController {
    @Autowired
    private SecondaryDisasterService secondaryDisasterService;

    @PostMapping("/secondDisaster/query")
    @ApiOperation(value = "次生灾害查询 @author 卓大")
    public ResponseDTO<PageResult<SecondaryDisasterVO>> query(@Valid @RequestBody SecondaryDisasterQueryForm query) {
        return secondaryDisasterService.querySecondaryDisaster(query);
    }

    @ApiOperation(value = "添加次生灾害 @author 卓大")
    @PostMapping("/secondary/add")
//    @PreAuthorize("@saAuth.checkPermission('system:employee:add')")
    public ResponseDTO<String> addSecondaryDisaster(@Valid @RequestBody SecondaryDisasterAddForm secondaryDisasterAddForm) {
        return secondaryDisasterService.addSecondaryDisaster(secondaryDisasterAddForm);
    }

    @ApiOperation(value = "更新次生灾害 @author 卓大")
    @PostMapping("/secondary/update")
//    @PreAuthorize("@saAuth.checkPermission('system:employee:update')")
    public ResponseDTO<String> updateSecondaryDisaster(@Valid @RequestBody SecondaryDisasterAddForm secondaryDisasterUpdateForm) {
        return secondaryDisasterService.updateSecondaryDisaster(secondaryDisasterUpdateForm);
    }


    @ApiOperation(value = "批量删除次生灾害 @author 卓大")
    @PostMapping("/secondary/update/batch/delete")
//    @PreAuthorize("@saAuth.checkPermission('system:employee:delete')")
    public ResponseDTO<String> batchUpdateDeleteFlag(@RequestBody List<String> IdList) {
        return secondaryDisasterService.batchUpdateDeleteFlag(IdList);
    }


    @ApiOperation("查询所有生命线工程 @author 卓大")
    @GetMapping("/secondary/queryAll")
    public ResponseDTO<List<SecondaryDisasterVO>> queryAllSecondaryDisaster() {
        return secondaryDisasterService.queryAllSecondaryDisaster();
    }

}