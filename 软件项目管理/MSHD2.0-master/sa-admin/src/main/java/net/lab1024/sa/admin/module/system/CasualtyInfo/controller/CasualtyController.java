package net.lab1024.sa.admin.module.system.CasualtyInfo.controller;

import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import net.lab1024.sa.admin.constant.AdminSwaggerTagConst;
import net.lab1024.sa.admin.module.system.CasualtyInfo.domain.form.CasualtyAddForm;
import net.lab1024.sa.admin.module.system.CasualtyInfo.domain.form.CasualtyQueryForm;
import net.lab1024.sa.admin.module.system.CasualtyInfo.domain.vo.CasualtyVO;
import net.lab1024.sa.admin.module.system.CasualtyInfo.service.CasualtyService;
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

@RestController
@OperateLog
@Api(tags = {AdminSwaggerTagConst.System.SYSTEM_SUPPORT_DISASTER})
public class CasualtyController {
    @Autowired
    private CasualtyService casualtyService;

    @PostMapping("/casualty/query")
    @ApiOperation(value = "伤亡信息查询")
    public ResponseDTO<PageResult<CasualtyVO>> query(@Valid @RequestBody CasualtyQueryForm query){
        return casualtyService.queryCasualty(query);
    }

    @ApiOperation(value = "添加伤亡信息 @author 卓大")
    @PostMapping("/casualty/add")
//    @PreAuthorize("@saAuth.checkPermission('system:employee:add')")
    public ResponseDTO<String> addCasualty(@Valid @RequestBody CasualtyAddForm casualtyAddForm) {
        return casualtyService.addCasualty(casualtyAddForm);
    }

    @ApiOperation(value = "更新伤亡信息 @author 卓大")
    @PostMapping("/casualty/update")
//    @PreAuthorize("@saAuth.checkPermission('system:employee:update')")
    public ResponseDTO<String> updateCasualty(@Valid @RequestBody CasualtyAddForm casualtyUpdateForm) {
        return casualtyService.updateCasualty(casualtyUpdateForm);
    }


    @ApiOperation(value = "批量删除伤亡信息 @author 卓大")
    @PostMapping("/casualty/update/batch/delete")
//    @PreAuthorize("@saAuth.checkPermission('system:employee:delete')")
    public ResponseDTO<String> batchUpdateDeleteFlag(@RequestBody List<String> IdList) {
        return casualtyService.batchUpdateDeletedFlag(IdList);
    }


    @ApiOperation("查询所有伤亡信息 @author 卓大")
    @GetMapping("/casualty/queryAll")
    public ResponseDTO<List<CasualtyVO>> queryAllCasualty() {
        return casualtyService.queryAllCasualty();
    }

}
