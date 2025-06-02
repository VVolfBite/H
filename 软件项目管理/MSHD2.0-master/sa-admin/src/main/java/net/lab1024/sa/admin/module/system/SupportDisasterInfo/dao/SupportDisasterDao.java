package net.lab1024.sa.admin.module.system.SupportDisasterInfo.dao;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import net.lab1024.sa.admin.module.system.SupportDisasterInfo.domain.entity.SupportDisasterEntity;
import net.lab1024.sa.admin.module.system.SupportDisasterInfo.domain.form.SupportDisasterQueryForm;
import net.lab1024.sa.admin.module.system.SupportDisasterInfo.domain.vo.SupportDisasterVO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Component;

import java.util.Collection;
import java.util.List;

@Mapper
@Component
public interface SupportDisasterDao extends BaseMapper<SupportDisasterEntity> {
    /**
     * 查询生命线工程列表
     *
     * @param page
     * @param queryForm
     * @return
     */
    List<SupportDisasterVO> querySupportDisaster(Page page, @Param("queryForm") SupportDisasterQueryForm queryForm);

    /**
     * 查询生命线工程
     */
    List<SupportDisasterVO> selectSupportDisasterByDeleted(@Param("deletedFlag") Boolean deletedFlag);

    /**
     * 通过ID查询
     *
     * @param ID
     * @return
     */
    SupportDisasterEntity getByID(@Param("ID") String ID);

    /**
     * 通过发生的地点查询
     *
     * @param location
     * @return
     */
    SupportDisasterEntity getByLocation(@Param("Location") String location);


    /**
     * 通过受灾程度查询
     *
     * @param Extent
     * @return
     */
    SupportDisasterEntity getByExtent(@Param("Extent") String Extent);

    /**
     * 通过受灾类型查询
     *
     * @param Label
     * @return
     */
    SupportDisasterEntity getByLabel(@Param("Label") String Label);

    /**
     * 获取所有员工
     *
     * @return
     */
    List<SupportDisasterVO> listAll();


    /**
     * 获取一批员工
     *
     * @param IDs
     * @return
     */
    List<SupportDisasterVO> getSupportDisasterByIds(@Param("IDs") Collection<String> IDs);


    /**
     * 查询单个员工信息
     *
     * @param ID
     * @return
     */
    SupportDisasterVO getSupportDisasterById(@Param("ID") String ID);

    /**
     * 获取所有
     *
     * @param leaveFlag
     * @return
     */

}
