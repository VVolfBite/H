package net.lab1024.sa.admin.module.system.SupportDisasterInfo.controller;

import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import net.lab1024.sa.admin.common.AdminBaseController;
import net.lab1024.sa.admin.constant.AdminSwaggerTagConst;
import net.lab1024.sa.admin.module.system.SupportDisasterInfo.domain.form.SupportDisasterAddForm;
import net.lab1024.sa.admin.module.system.SupportDisasterInfo.domain.form.SupportDisasterQueryForm;
import net.lab1024.sa.admin.module.system.SupportDisasterInfo.domain.vo.SupportDisasterVO;
import net.lab1024.sa.admin.module.system.SupportDisasterInfo.service.SupportDisasterService;
import net.lab1024.sa.common.common.domain.PageResult;
import net.lab1024.sa.common.common.domain.ResponseDTO;
import net.lab1024.sa.common.module.support.operatelog.annoation.OperateLog;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.util.List;

/**
 *  生命线工程
 */

@RestController
@OperateLog
@Api(tags = {AdminSwaggerTagConst.System.SYSTEM_SUPPORT_DISASTER})
public class SupportDisasterController extends AdminBaseController {
    @Autowired
    private SupportDisasterService supportDisasterService;

    @PostMapping("/support/query")
    @ApiOperation(value = "生命线工程管理查询 @author 卓大")
    public ResponseDTO<PageResult<SupportDisasterVO>> query(@Valid @RequestBody SupportDisasterQueryForm query) {
        return supportDisasterService.querySupportDisaster(query);
    }

    @ApiOperation(value = "添加生命线工程 @author 卓大")
    @PostMapping("/support/add")
//    @PreAuthorize("@saAuth.checkPermission('system:employee:add')")
    public ResponseDTO<String> addSupportDisaster(@Valid @RequestBody SupportDisasterAddForm supportDisasterAddForm) {
        return supportDisasterService.addSupportDisaster(supportDisasterAddForm);
    }

    @ApiOperation(value = "更新生命线工程 @author 卓大")
    @PostMapping("/support/update")
//    @PreAuthorize("@saAuth.checkPermission('system:employee:update')")
    public ResponseDTO<String> updateSupportDisaster(@Valid @RequestBody SupportDisasterAddForm supportDisasterUpdateForm) {
        return supportDisasterService.updateSupportDisaster(supportDisasterUpdateForm);
    }


    @ApiOperation(value = "批量删除生命线工程 @author 卓大")
    @PostMapping("/support/update/batch/delete")
//    @PreAuthorize("@saAuth.checkPermission('system:employee:delete')")
    public ResponseDTO<String> batchUpdateDeleteFlag(@RequestBody List<String> IdList) {
        return supportDisasterService.batchUpdateDeleteFlag(IdList);
    }


    @ApiOperation("查询所有生命线工程 @author 卓大")
    @GetMapping("/support/queryAll")
    public ResponseDTO<List<SupportDisasterVO>> queryAllSupportDisaster() {
        return supportDisasterService.queryAllSupportDisaster();
    }

}
